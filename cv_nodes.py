import math
import cv2 as cv
import numpy as np

from .dry import *
from .color_utils import *

# TODO these nodes return the mask, not the image with the background removed!
#       this is somewhat misleading. Consider changing the methods names.
#       ( but to what? GrabCutMask? FramedMaskGrabCutMask? ...)

# region types

thresh_types_map = {
    'BINARY': cv.THRESH_BINARY,
    'BINARY_INV': cv.THRESH_BINARY_INV,
    'TRUNC': cv.THRESH_TRUNC,
    'TOZERO': cv.THRESH_TOZERO,
    'TOZERO_INV': cv.THRESH_TOZERO_INV,
}
thresh_types = list(thresh_types_map.keys())

border_types_map = {
    'BORDER_CONSTANT': cv.BORDER_CONSTANT,
    'BORDER_REPLICATE': cv.BORDER_REPLICATE,
    'BORDER_REFLECT': cv.BORDER_REFLECT,
    'BORDER_REFLECT101': cv.BORDER_REFLECT101,
    'BORDER_WRAP': cv.BORDER_WRAP,
    'BORDER_TRANSPARENT': cv.BORDER_TRANSPARENT,
    'BORDER_REFLECT_101': cv.BORDER_REFLECT_101,
    'BORDER_DEFAULT': cv.BORDER_DEFAULT,
    'BORDER_ISOLATED': cv.BORDER_ISOLATED
}

border_types = list(border_types_map.keys())

border_types_excluding_transparent = border_types_map.copy()
border_types_excluding_transparent.pop("BORDER_TRANSPARENT")
border_types_excluding_transparent = list(border_types_excluding_transparent.keys())


# endregion


# region misc

class CopyMakeBorderSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "border_size": ("INT", {"default": 64}),
                "border_type": (border_types_excluding_transparent, border_types[0])
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_border"
    CATEGORY = "Bmad/CV"

    def make_border(self, image, border_size, border_type):
        image = tensor2opencv(image, 0)
        image = cv.copyMakeBorder(image, border_size, border_size, border_size, border_size,
                                  border_types_map[border_type])
        image = opencv2tensor(image)
        return (image,)


class ConvertImg:
    """ An explicit conversion, instead of using workarounds when using certain custom nodes. """
    options_map = {
        "RGBA": 4,
        "RGB": 3,
        "GRAY": 1,
    }
    options = list(options_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "to": (s.options, {"default": s.options[1]})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "Bmad/CV"

    def convert(self, image, to):
        image = tensor2opencv(image, self.options_map[to])
        return (opencv2tensor(image),)


class AddAlpha:
    method = ["default", "invert"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
            },
            "optional": {
                "alpha": ("IMAGE",),
                "method": (s.method, {"default": s.method[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_alpha"
    CATEGORY = "Bmad/image"

    def add_alpha(self, rgb_image, alpha=None, method=None):
        rgb_image = tensor2opencv(rgb_image, 3)
        rgba = cv.cvtColor(rgb_image, cv.COLOR_RGB2RGBA)
        if alpha is not None:
            alpha = tensor2opencv(alpha, 1)
            rgba[:, :, 3] = alpha if method == self.method[0] else 255 - alpha
        rgba = opencv2tensor(rgba)
        return (rgba,)


class FadeMaskEdges:
    """
    The original intent is to premultiply and alpha blend a subject's edges to avoid outer pixels creeping in.

    A very slight blur near the edges afterwards when using paste_original_blacks and low tightness may be required,
     but this should be done after premultiplying and setting the alpha.

    Stylized subject's, such as drawings with black outlines, may benefit from using different 2 edge fades:
        1. a fade with higher edge size for the premultiplication, fading the subject into blackness
        2. a tighter fade for the alpha
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "edge_size": ("FLOAT", {"default": 5.0, "min": 1.0, "step": 1.0}),
                # how quick does it fade to black
                "edge_tightness": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 10.0, "step": 0.05}),
                # how does it fade, may be used to weaken small lines; 1 = linear transition
                "edge_exponent": ("FLOAT", {"default": 1, "min": 0.1, "max": 10.0, "step": 0.1}),
                "smoothing_diameter": ("INT", {"default": 10, "min": 2, "max": 256, "step": 1}),
                "paste_original_blacks": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Bmad/CV"

    def apply(self, binary_image, edge_size, edge_tightness, edge_exponent, smoothing_diameter, paste_original_blacks):
        binary_image = tensor2opencv(binary_image, 1)
        # _, binary_image = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY) # suppose it's already binary

        # compute L2 (euclidean) distance -> normalize with respect to edge size -> smooth
        distance_transform = cv.distanceTransform(binary_image, cv.DIST_L2, cv.DIST_MASK_3)
        normalized_distance = distance_transform / edge_size
        smoothed_distance = cv.bilateralFilter(normalized_distance, smoothing_diameter, 75, 75)

        # darken the white pixels based on smoothed distance and "edge tightness"
        diff = 1 - smoothed_distance
        darkened_image = (abs(diff*edge_tightness) ** (1/edge_exponent)) * np.sign(diff)
        darkened_image = np.clip(darkened_image, 0, 1)
        darkened_image = (darkened_image * 255).astype(np.uint8)

        if paste_original_blacks:  # mask original black pixels
            black_mask = binary_image < 1
            darkened_image[black_mask] = 0

        output_image = binary_image - darkened_image  # darken original image
        output_image = opencv2tensor(output_image)
        return (output_image, )


# endregion


# region grabcut nodes

class FramedMaskGrabCut:
    frame_options_values = {
        'FULL_FRAME': 0,
        'IGNORE_BOTTOM': 1,
        'IGNORE_TOP': 2,
        'IGNORE_RIGHT': 4,
        'IGNORE_LEFT': 8,
        'IGNORE_HORIZONTAL': 12,
        'IGNORE_VERTICAL': 3,
    }
    frame_options = list(frame_options_values.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "thresh": ("IMAGE",),
                "iterations": ("INT", {
                    "default": 25,
                    "min": 0,
                    "max": 200,
                    "step": 1
                }),
                "margin": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "frame_option": (s.frame_options, {
                    "default": s.frame_options[0]
                }),

                # to only use PR FGD set threshold_FGD to 0
                # to only use only FGD set threshold_FGD to a lower value than threshold_PR_FGD
                # using one of these also works as a safeguard in case thresh has other values besides 0s and 1s
                "threshold_FGD": ("INT", {
                    "default": 250,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "threshold_PR_FGD": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 255,
                    "step": 1
                }),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grab_cut"
    CATEGORY = "Bmad/CV/GrabCut"

    def grab_cut(self, image, thresh, iterations, margin, frame_option, threshold_FGD, threshold_PR_FGD, output_format):
        image = tensor2opencv(image)
        thresh = tensor2opencv(thresh, 1)

        assert image.shape[:2] == thresh.shape

        fg_model = np.zeros((1, 65), dtype="float")
        bg_model = np.zeros((1, 65), dtype="float")
        mask = np.full(image.shape[:2], cv.GC_PR_BGD, dtype=np.uint8)  # probable background
        # foreground and probable foreground
        if threshold_FGD > threshold_PR_FGD:
            mask[thresh >= threshold_PR_FGD] = cv.GC_PR_FGD
        if threshold_FGD > 0:
            mask[thresh >= threshold_FGD] = cv.GC_FGD

        # check what borders should be painted
        frame_option = self.frame_options_values[frame_option]
        include_bottom = not (frame_option & self.frame_options_values['IGNORE_BOTTOM'])
        include_top = not (frame_option & self.frame_options_values['IGNORE_TOP'])
        include_right = not (frame_option & self.frame_options_values['IGNORE_RIGHT'])
        include_left = not (frame_option & self.frame_options_values['IGNORE_LEFT'])

        # paint the borders as being background
        if include_bottom:
            mask[-margin:, :] = cv.GC_BGD
        if include_top:
            mask[0:margin, :] = cv.GC_BGD
        if include_right:
            mask[:, -margin:] = cv.GC_BGD
        if include_left:
            mask[:, 0:margin] = cv.GC_BGD

        mask, bg_model, fg_model = cv.grabCut(image, mask, None, bg_model, fg_model, iterCount=iterations,
                                              mode=cv.GC_INIT_WITH_MASK)

        # generate mask with "pixels" classified as background/foreground
        output_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1)

        output_mask = (output_mask * 255).astype("uint8")

        output_mask = maybe_convert_img(output_mask, 1, image_output_formats_options_map[output_format])
        image = opencv2tensor(output_mask)

        return (image,)


class RectGrabCut:
    # TODO add option to crop or just leave as 0 the section outside the rect
    # TODO maybe add option to exclude PR_BGD or include PR_FGD in outputMask

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "x1": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "y1": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "x2": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "y2": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 2000,
                    "step": 1
                }),
                "iterations": ("INT", {
                    "default": 25,
                    "min": 0,
                    "max": 200,
                    "step": 1
                }),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grab_cut"

    CATEGORY = "Bmad/CV/GrabCut"

    def grab_cut(self, image, iterations, x1, y1, x2, y2, output_format):
        image = tensor2opencv(image)

        fg_model = np.zeros((1, 65), dtype="float")
        bg_model = np.zeros((1, 65), dtype="float")
        mask = np.zeros(image.shape[:2], dtype="uint8")
        rect = (x1, y1, x2, y2)

        mask, bg_model, fg_model = cv.grabCut(image, mask, rect, bg_model,
                                              fg_model, iterCount=iterations, mode=cv.GC_INIT_WITH_RECT)

        # generate mask with "pixels" classified as background/foreground
        output_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),
                               0, 1)
        output_mask = (output_mask * 255).astype("uint8")

        output_mask = maybe_convert_img(output_mask, 1, image_output_formats_options_map[output_format])
        image = opencv2tensor(output_mask)
        # image = image[y1:y2, x1:x2] #TODO maybe add option whether to crop or not

        return (image,)


class FramedMaskGrabCut2:
    # TODO option to ignore probable background in sure_thresh

    frame_options = ['FULL_FRAME', 'IGNORE_BOTTOM', 'IGNORE_TOP', 'IGNORE_RIGHT', 'IGNORE_LEFT', 'IGNORE_HORIZONTAL'
        , 'IGNORE_VERTICAL']
    frame_options_values = {
        'FULL_FRAME': 0,
        'IGNORE_BOTTOM': 1,
        'IGNORE_TOP': 2,
        'IGNORE_RIGHT': 4,
        'IGNORE_LEFT': 8,
        'IGNORE_HORIZONTAL': 12,
        'IGNORE_VERTICAL': 3,
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "thresh_maybe": ("IMAGE",),
                "thresh_sure": ("IMAGE",),
                "iterations": ("INT", {
                    "default": 25,
                    "min": 0,
                    "max": 200,
                    "step": 1
                }),
                "margin": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "frame_option": (s.frame_options, {
                    "default": 'FULL_FRAME'
                }),
                # source thresh may not be only 0s and 1s, use this as a safeguard
                "binary_threshold": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 255,
                    "step": 1
                }),
                "maybe_black_is_sure_background": ("BOOLEAN", {"default": False}),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grab_cut"

    CATEGORY = "Bmad/CV/GrabCut"

    def grab_cut(self, image, thresh_maybe, thresh_sure, iterations,
                 margin, frame_option, binary_threshold,
                 maybe_black_is_sure_background, output_format):
        image = tensor2opencv(image)

        thresh_maybe = tensor2opencv(thresh_maybe, 1)
        thresh_sure = tensor2opencv(thresh_sure, 1)

        fg_model = np.zeros((1, 65), dtype="float")
        bg_model = np.zeros((1, 65), dtype="float")
        mask = np.full(image.shape[:2], cv.GC_PR_BGD, dtype=np.uint8)  # probable background
        mask[thresh_maybe >= binary_threshold] = cv.GC_PR_FGD  # probable foreground
        mask[thresh_sure >= binary_threshold] = cv.GC_FGD  # foreground

        frame_option = self.frame_options_values[frame_option]
        include_bottom = not (frame_option & self.frame_options_values['IGNORE_BOTTOM'])
        include_top = not (frame_option & self.frame_options_values['IGNORE_TOP'])
        include_right = not (frame_option & self.frame_options_values['IGNORE_RIGHT'])
        include_left = not (frame_option & self.frame_options_values['IGNORE_LEFT'])

        if include_bottom:
            mask[-margin:, :] = cv.GC_BGD
        if include_top:
            mask[0:margin, :] = cv.GC_BGD
        if include_right:
            mask[:, -margin:] = cv.GC_BGD
        if include_left:
            mask[:, 0:margin] = cv.GC_BGD

        if maybe_black_is_sure_background:
            mask[thresh_maybe < binary_threshold] = cv.GC_BGD  # background

        mask, bg_model, fg_model = cv.grabCut(image, mask, None, bg_model, fg_model, iterCount=iterations,
                                              mode=cv.GC_INIT_WITH_MASK)

        # generate mask with "pixels" classified as background/foreground
        output_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1)
        output_mask = (output_mask * 255).astype("uint8")

        output_mask = maybe_convert_img(output_mask, 1, image_output_formats_options_map[output_format])
        image = opencv2tensor(output_mask)

        return (image,)


# endregion grabcut nodes


# region contour nodes

class Contours:
    """
    Note:
    The image is converted to grey, but no threshold is applied.
    Apply the thresholding before using and feed a black and white image.
    """

    approximation_modes_map = {
        'CHAIN_APPROX_NONE': cv.CHAIN_APPROX_NONE,
        'CHAIN_APPROX_SIMPLE': cv.CHAIN_APPROX_SIMPLE,
        'CHAIN_APPROX_TC89_L1': cv.CHAIN_APPROX_TC89_L1,
        'CHAIN_APPROX_TC89_KCOS': cv.CHAIN_APPROX_TC89_KCOS
    }
    approximation_modes = list(approximation_modes_map.keys())

    retrieval_modes_map = {
        'RETR_EXTERNAL': cv.RETR_EXTERNAL,
        'RETR_LIST': cv.RETR_LIST,
        'RETR_CCOMP': cv.RETR_CCOMP,
        'RETR_TREE': cv.RETR_TREE,
        'RETR_FLOODFILL': cv.RETR_FLOODFILL
    }
    retrieval_modes = list(retrieval_modes_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "retrieval_mode": (s.retrieval_modes, {"default": "RETR_LIST"}),
                "approximation_mode": (s.approximation_modes, {"default": "CHAIN_APPROX_SIMPLE"}),
            },
        }

    RETURN_TYPES = ("CV_CONTOUR", "CV_CONTOURS_HIERARCHY")
    FUNCTION = "find_contours"
    CATEGORY = "Bmad/CV/Contour"
    OUTPUT_IS_LIST = (True, False)

    def find_contours(self, image, retrieval_mode, approximation_mode):
        image = tensor2opencv(image)
        thresh = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # no thresh applied here, non zeroes are treated as 1 according to documentation;
        # thresh should have been already applied to the image, before passing it to this node.

        return cv.findContours(
            thresh,
            self.retrieval_modes_map[retrieval_mode],
            self.approximation_modes_map[approximation_mode])


class DrawContours:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "contours": ("CV_CONTOUR",),
                "index_to_draw": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 1000,
                    "step": 1
                }),
                "thickness": ("INT", {
                    "default": 5,
                    "min": -1,
                    "max": 32,
                    "step": 1
                }),
                "color": ("COLOR",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"

    CATEGORY = "Bmad/CV/Contour"
    INPUT_IS_LIST = True

    def draw(self, image, contours, index_to_draw, color, thickness):
        image = image[0]
        index_to_draw = index_to_draw[0]
        color = color[0]
        thickness = thickness[0]

        background = tensor2opencv(image)

        um_image = cv.UMat(background)
        cv.drawContours(um_image, contours, index_to_draw, ImageColor.getcolor(color, "RGB"), thickness)
        contour_image = um_image.get()

        image = opencv2tensor(contour_image)

        return (image,)


class GetContourFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "contours": ("CV_CONTOUR",),
                "index": ("INT", {"default": 0, "min": 0, "step": 1})
            }
        }

    RETURN_TYPES = ("CV_CONTOUR",)
    FUNCTION = "get_contour"
    CATEGORY = "Bmad/CV/Contour"
    INPUT_IS_LIST = True

    def get_contour(self, contours, index):
        index = index[0]
        if index >= len(contours):
            return (None,)
        return (contours[index],)


class ContourGetBoundingRect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "contour": ("CV_CONTOUR",),
                "return_mode": (rect_modes, {"default": rect_modes[1]})
            },
        }

    RETURN_TYPES = tuple(["INT" for x in range(4)])
    FUNCTION = "compute"
    CATEGORY = "Bmad/CV/Contour"

    def compute(self, contour, return_mode):
        if contour is None:
            print("Contour = None !")
            return (0, 0, 0, 0,)

        # convert opencv boundingRect format to bounds
        bounds = rect_modes_map[rect_modes[0]]["toBounds"](*cv.boundingRect(contour))

        # convert from bounds to desired output format on return
        return rect_modes_map[return_mode]["fromBounds"](*bounds)


class FilterContour:
    def MODE(self, cnts, fit):
        sorted_list = sorted(cnts, key=fit)
        return [sorted_list[len(sorted_list) // 2]]

    return_modes_map = {
        "MAX": lambda cnts, fit: [sorted(cnts, key=fit)[-1]],
        "MIN": lambda cnts, fit: [sorted(cnts, key=fit)[0]],
        "MODE": MODE,
        "FILTER": lambda cnts, fit: filter(fit, cnts),
    }
    return_modes = list(return_modes_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "contours": ("CV_CONTOUR",),
                "fitness": ("STRING", {"multiline": True, "default":
                    "# Contour Fitness Function\n"}),
                "select": (s.return_modes, {"default": s.return_modes[0]})
            },
            "optional": {
                "image": ("IMAGE",),
                "aux_contour": ("CV_CONTOUR",)
            }
        }

    RETURN_TYPES = ("CV_CONTOUR",)
    FUNCTION = "filter"
    CATEGORY = "Bmad/CV/Contour"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def filter(self, contours, fitness, select, image=None, aux_contour=None):
        import math
        import cv2
        import numpy

        fitness = fitness[0]
        select = select[0]

        # region prepare inputs
        if image is not None:
            image = image[0]
            image = tensor2opencv(image)
        if aux_contour is not None:
            aux_contour = aux_contour[0]

        fitness = prepare_text_for_eval(fitness)

        # endregion

        # region available functions
        # cv methods, but cache them
        @cache_with_ids(single=False)
        def boundingRect(cnt):
            return cv.boundingRect(cnt)

        @cache_with_ids(single=False)
        def contourArea(cnt):
            return cv.contourArea(cnt)

        @cache_with_ids(single=False)
        def arcLength(cnt):
            return cv.arcLength(cnt, True)

        @cache_with_ids(single=True)
        def minAreaRect(cnt):
            return cv.minAreaRect(cnt)

        @cache_with_ids(single=True)
        def minEnclosingCircle(cnt):
            return cv.minEnclosingCircle(cnt)

        @cache_with_ids(single=True)
        def fitEllipse(cnt):
            return cv.fitEllipse(cnt)

        @cache_with_ids(single=True)
        def convexHull(cnt):
            return cv.convexHull(cnt)

        # useful properties; adapted from multiple sources, including cv documentation
        @cache_with_ids(single=True)
        def aspect_ratio(cnt):
            x, y, w, h = boundingRect(cnt)
            return float(w) / h

        @cache_with_ids(single=True)
        def extent(cnt):
            area = contourArea(cnt)
            x, y, w, h = boundingRect(cnt)
            rect_area = w * h
            return float(area) / rect_area

        @cache_with_ids(single=True)
        def solidity(cnt):
            area = contourArea(cnt)
            hull = convexHull(cnt)
            hull_area = contourArea(hull)
            return float(area) / hull_area

        @cache_with_ids(single=True)
        def equi_diameter(cnt):
            area = contourArea(cnt)
            return math.sqrt(4 * area / math.pi)

        @cache_with_ids(single=True)
        def center(cnt):
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY

        @cache_with_ids(single=False)
        def contour_mask(cnt, img):
            if len(img.shape) > 2:
                height, width, channels = img.shape
            else:
                height, width = img.shape

            mask = numpy.zeros((height, width, 1), numpy.uint8)
            cv.drawContours(mask, [cnt], 0, 255, -1)
            return mask

        @cache_with_ids(single=True)
        def mean_color(cnt, img):
            return cv.mean(img, mask=contour_mask(cnt, img))

        @cache_with_ids(single=True)
        def mean_intensity(cnt, img):
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            return mean_color(cnt, gray)[0]

        @cache_with_ids(single=True)
        def extreme_points(cnt):
            l = tuple(cnt[cnt[:, :, 0].argmin()][0])
            r = tuple(cnt[cnt[:, :, 0].argmax()][0])
            t = tuple(cnt[cnt[:, :, 1].argmin()][0])
            b = tuple(cnt[cnt[:, :, 1].argmax()][0])
            return {"top": t, "right": r, "bottom": b, "left": l}

        def intercepts_mask(cnt, img):  # where img should be a binary mask
            gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            intersection = cv2.bitwise_and(gray,
                                           cv2.drawContours(np.zeros_like(gray), [cnt], 0, 255, thickness=cv2.FILLED))
            return cv2.countNonZero(intersection) > 0

        # endregion

        available_funcs = {}
        for key, value in locals().items():
            if callable(value):
                available_funcs[key] = value

        fitness = eval(f"lambda c, i, a: {fitness}", {
            "__builtins__": {},
            "tuple": tuple, "list": list,
            'm': math, 'cv': cv2, 'np': numpy,
            **available_funcs
        }, {})

        return (self.return_modes_map[select]
                (contours, lambda c: fitness(c, image, aux_contour))
                ,)


class ContourToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "contour": ("CV_CONTOUR",),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "Bmad/CV/Contour"

    def draw(self, image, contour, output_format):
        image = tensor2opencv(image, 1)
        image = np.zeros(image.shape, dtype=np.uint8)
        cv.drawContours(image, [contour], 0, (255), -1)
        image = maybe_convert_img(image, 1, image_output_formats_options_map[output_format])
        image = opencv2tensor(image)
        return (image,)


# endregion contour nodes


# region Computational Photography

class SeamlessClone:
    clone_modes_map = {
        "NORMAL": cv.NORMAL_CLONE,
        "MIXED": cv.MIXED_CLONE,
        "MONO": cv.MONOCHROME_TRANSFER
    }
    clone_modes = list(clone_modes_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dst": ("IMAGE",),
                "src": ("IMAGE",),
                "src_mask": ("IMAGE",),
                "flag": (s.clone_modes, {"default": s.clone_modes[0]}),
                "cx": ("INT", {"default": 0, "min": -999999, "step": 1}),
                "cy": ("INT", {"default": 0, "min": -999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"
    CATEGORY = "Bmad/CV/C.Photography"

    def paste(self, src, dst, src_mask, flag, cx, cy):
        src = tensor2opencv(src)
        dst = tensor2opencv(dst)
        src_mask = tensor2opencv(src_mask, 1)

        result = cv.seamlessClone(src, dst, src_mask, (cx, cy), self.clone_modes_map[flag])
        result = opencv2tensor(result)

        return (result,)


class SeamlessCloneSimpler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dst": ("IMAGE",),
                "src": ("IMAGE",),
                "src_mask": ("IMAGE",),
                "flag": (SeamlessClone.clone_modes, {"default": SeamlessClone.clone_modes[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"
    CATEGORY = "Bmad/CV/C.Photography"

    @staticmethod
    def get_center(cv_mask):
        br = cv.boundingRect(cv_mask)
        return br[0] + br[2] // 2, br[1] + br[3] // 2

    def paste(self, src, dst, src_mask, flag):
        src_mask_cv = tensor2opencv(src_mask, 1)
        cx, cy = SeamlessCloneSimpler.get_center(src_mask_cv)
        sc = SeamlessClone()
        return sc.paste(src, dst, src_mask, flag, cx, cy)


class Inpaint:
    inpaint_method_map = {
        "TELEA": cv.INPAINT_TELEA,
        "NS": cv.INPAINT_NS,
    }
    inpaint_methods = list(inpaint_method_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "img": ("IMAGE",),
                "mask": ("IMAGE",),
                "radius": ("INT", {"default": 3, "min": 0, "step": 1}),
                "flag": (s.inpaint_methods, {"default": s.inpaint_methods[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paint"
    CATEGORY = "Bmad/CV/C.Photography"

    def paint(self, img, mask, radius, flag):
        img = tensor2opencv(img)
        mask = tensor2opencv(mask, 1)
        dst = cv.inpaint(img, mask, radius, self.inpaint_method_map[flag])
        result = opencv2tensor(dst)
        return (result,)


class ChameleonMask:  # wtf would I name this node as?
    mode_func_map = {
        "GRAY": lambda i: cv.cvtColor(i, cv.COLOR_BGR2GRAY),
        "VALUE": lambda i: cv.cvtColor(i, cv.COLOR_RGB2HSV)[:, :, 2],
        "LIGHTNESS": lambda i: cv.cvtColor(i, cv.COLOR_RGB2HLS)[:, :, 1],

        # not sure if these would be useful, but costs nothing to leave them here
        "HUE": lambda i: cv.cvtColor(i, cv.COLOR_RGB2HSV)[:, :, 0],
        "SATURATION (HSV)": lambda i: cv.cvtColor(i, cv.COLOR_RGB2HSV)[:, :, 1],
        "SATURATION (HSL)": lambda i: cv.cvtColor(i, cv.COLOR_RGB2HLS)[:, :, 2],
    }
    modes = list(mode_func_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dst": ("IMAGE",),
                "src": ("IMAGE",),
                "thresh_blur": ("INT", {"default": 30, "min": 2, "step": 2}),
                "close_dist": ("INT", {"default": 32, "min": 0, "step": 1}),
                "open_dist": ("INT", {"default": 32, "min": 0, "step": 1}),
                "size_dist": ("INT", {"default": 8, "min": -99999, "step": 1}),
                "mask_blur": ("INT", {"default": 64, "min": 0, "step": 2}),
                "contrast_adjust": ("FLOAT", {"default": 2.4, "min": 0, "max": 20, "step": .5}),
                "mode": (s.modes, {"default": s.modes[0]}),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                }),

            },
            "optional": {
                "optional_roi_mask": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    FUNCTION = "create_mask"
    CATEGORY = "Bmad/CV/C.Photography"

    def create_mask(self, src, dst, thresh_blur, close_dist, open_dist, size_dist, mask_blur,
                    contrast_adjust, mode, output_format, optional_roi_mask=None):
        src = tensor2opencv(src)
        dst = tensor2opencv(dst)

        thresh_blur += 1
        if mask_blur > 0:
            mask_blur += 1

        # compute the difference between images based on mode
        src = self.mode_func_map[mode](src)
        dst = self.mode_func_map[mode](dst)
        diff = cv.absdiff(src, dst)

        if mode == "HUE":
            diff = np.minimum(diff, 180 - diff)

        # binary thresholding
        # _, mask = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
        diff = cv.GaussianBlur(diff, (thresh_blur, thresh_blur), 0)
        ret3, mask = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        if optional_roi_mask is not None:
            optional_roi_mask = tensor2opencv(optional_roi_mask, 1)
            mask[optional_roi_mask < 127] = 0

        # morphological closing > closing > dilate/erode
        if close_dist > 0:
            close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_dist, close_dist))
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel)
        if open_dist > 0:
            open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_dist, open_dist))
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, open_kernel)

        if size_dist > 0:
            size_op = cv.MORPH_DILATE
            size = size_dist
        else:
            size_op = cv.MORPH_ERODE
            size = abs(size_dist)
        if size_dist != 0:
            size_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size))
            mask = cv.morphologyEx(mask, size_op, size_kernel)

        # gaussian blur + contrast adjust
        if mask_blur > 0:
            mask = cv.GaussianBlur(mask, (mask_blur, mask_blur), 0)
        mask = cv.convertScaleAbs(mask, alpha=1 + contrast_adjust, beta=0)  # / 100, beta=0)

        # convert to target format and output as tensor
        # note: diff is only meant to be used for debug purposes
        mask = maybe_convert_img(mask, 1, image_output_formats_options_map[output_format])
        diff = opencv2tensor(diff)
        mask = opencv2tensor(mask)

        return (mask, diff,)


# endregion Computational Photography


# region thresholding and eq

class OtsuThreshold:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # "channel": (s.channels, {"default": "greyscale"}),
                "threshold_type": (thresh_types, {"default": thresh_types[0]}),
                "gaussian_blur_x": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 200,
                    "step": 2
                }),
                "gaussian_blur_y": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 200,
                    "step": 2
                }),
                "gaussian_border_type": (border_types, {"default": border_types[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "otsu_thresthold"
    CATEGORY = "Bmad/CV/Thresholding"

    def otsu_thresthold(self, image, threshold_type, gaussian_blur_x, gaussian_blur_y, gaussian_border_type):
        image = tensor2opencv(image, 1)
        if gaussian_blur_x > 0 and gaussian_blur_y > 0:
            image = cv.GaussianBlur(image, (gaussian_blur_x + 1, gaussian_blur_y + 1),
                                    border_types_map[gaussian_border_type])
        _, image = cv.threshold(image, 0, 255, thresh_types_map[threshold_type] + cv.THRESH_OTSU)
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        image = opencv2tensor(image)
        return (image,)


class AdaptiveThresholding:
    adaptive_modes_map = {
        "ADAPTIVE_THRESH_MEAN_C": cv.ADAPTIVE_THRESH_MEAN_C,
        "ADAPTIVE_THRESH_GAUSSIAN_C": cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    }
    adaptive_modes = list(adaptive_modes_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",),
                "max_value": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                # maybe should just allow for 255? may just confuse some people that don't read documentation
                "adaptive_method": (s.adaptive_modes, {"default": s.adaptive_modes[1]}),
                "threshold_type": (thresh_types, {"default": thresh_types[0]}),
                "block_size": ("INT", {"default": 4, "min": 2, "step": 2}),
                "c": ("INT", {"default": 2, "min": -999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "thresh"
    CATEGORY = "Bmad/CV/Thresholding"

    def thresh(self, src, max_value, adaptive_method, threshold_type, block_size, c):
        # maybe allow to use from a specific channel 1st? nah, just create a node to fetch the channel
        # might be useful for other nodes
        src = tensor2opencv(src, 1)
        src = cv.adaptiveThreshold(src, max_value, self.adaptive_modes_map[adaptive_method], \
                                   thresh_types_map[threshold_type], block_size + 1, c)
        src = cv.cvtColor(src, cv.COLOR_GRAY2RGB)
        src = opencv2tensor(src)
        return (src,)


class EqualizeHistogram:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "eq"
    CATEGORY = "Bmad/CV/Thresholding"

    def eq(self, src):
        src = tensor2opencv(src, 1)
        eq = cv.equalizeHist(src)
        eq = cv.cvtColor(eq, cv.COLOR_GRAY2RGB)
        eq = opencv2tensor(eq)
        return (eq,)


class CLAHE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",),
                "clip_limit": ("INT", {"default": 2, "step": 1}),
                # 40 is the default in documentation, but prob. a bit high no?
                "tile_grid_x": ("INT", {"default": 8, "min": 2, "step": 1}),
                "tile_grid_y": ("INT", {"default": 8, "min": 2, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "eq"
    CATEGORY = "Bmad/CV/Thresholding"

    def eq(self, src, clip_limit, tile_grid_x, tile_grid_y):
        src = tensor2opencv(src, 1)
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_x, tile_grid_y))
        eq = clahe.apply(src)
        eq = cv.cvtColor(eq, cv.COLOR_GRAY2RGB)
        eq = opencv2tensor(eq)
        return (eq,)


class FindThreshold:
    """
    simple cond examples:
        cv.countNonZero(t)  > 100  # the number of non black pixels (white when using binary thresh type)
        (t.size - cv.countNonZero(t)) / t.size > .50 # the percentage of black pixels is higher than 50%
        # TODO can detect some shape(s) (requires optional inputs, and for current output maybe not that useful
    if none is found, returns the last
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",),
                "start_at": ("INT", {"default": 1, "min": 1, "max": 255, "step": 1}),
                "end_at": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1}),
                "thresh_type": (thresh_types, {"default": thresh_types[0]}),
                "downscale_factor": ("INT", {"default": 2, "min": 1, "step": 1}),
                "condition": ("STRING", {"multiline": True, "default":
                    "# Some expression that returns True or False\n"}),
            },
        }  # TODO optional inputs

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "search"
    CATEGORY = "Bmad/CV/Thresholding"

    def search(self, src, start_at, end_at, thresh_type, downscale_factor, condition):
        import cv2
        import numpy
        import math

        o_img = tensor2opencv(src, 1)
        height, width = tuple(o_img.shape)
        img = cv.resize(o_img, (height // downscale_factor, width // downscale_factor), interpolation=cv.INTER_AREA)

        max_v = max(start_at, end_at)
        min_v = min(start_at, end_at)
        range_to_check = range(min_v, max_v + 1)
        if end_at < start_at:
            range_to_check = range_to_check.__reversed__()

        condition = prepare_text_for_eval(condition)
        cond_check = eval(f"lambda t: {condition}", {
            "__builtins__": {},
            "tuple": tuple, "list": list,
            'm': math, 'cv': cv2, 'np': numpy
        }, {})

        thresh_value = end_at
        for i in range_to_check:
            _, thresh = cv.threshold(img, i, 255, thresh_types_map[thresh_type])
            if cond_check(thresh):
                thresh_value = i
                break

        _, img = cv.threshold(o_img, thresh_value, 255, thresh_types_map[thresh_type])
        img = opencv2tensor(img)
        return (img,)


class InRangeHSV:
    # w/ respect to documentation in :
    #   https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
    # both bounds are inclusive

    @staticmethod
    def get_saturation_and_value_bounds(color_a, color_b):
        min_s = min(color_a[1], color_b[1])
        max_s = max(color_a[1], color_b[1])
        min_v = min(color_a[2], color_b[2])
        max_v = max(color_a[2], color_b[2])
        return min_s, max_s, min_v, max_v

    @staticmethod
    def hue_ignore(image, color_a, color_b):
        ls, us, lv, uv = InRangeHSV.get_saturation_and_value_bounds(color_a, color_b)
        return cv.inRange(image, np.array((0, ls, lv)), np.array((179, us, uv)))

    @staticmethod
    def hue_single(image, color_a, color_b):
        ls, us, lv, uv = InRangeHSV.get_saturation_and_value_bounds(color_a, color_b)
        lh = min(color_a[0], color_b[0])
        uh = max(color_a[0], color_b[0])
        return cv.inRange(image, np.array((lh, ls, lv)), np.array((uh, us, uv)))

    @staticmethod
    def hue_split(image, color_a, color_b):
        ls, us, lv, uv = InRangeHSV.get_saturation_and_value_bounds(color_a, color_b)
        lh = min(color_a[0], color_b[0])
        uh = max(color_a[0], color_b[0])
        thresh_1 = cv.inRange(image, np.array((0, ls, lv)), np.array((lh, us, uv)))
        thresh_2 = cv.inRange(image, np.array((uh, ls, lv)), np.array((179, us, uv)))
        return cv.bitwise_or(thresh_1, thresh_2)

    LARGEST_HUE_INTERVAL = False
    SMALLEST_HUE_INTERVAL = True

    @staticmethod
    def choose_hue_method(color_a, color_b, interval_to_select):
        single_interval = abs(color_a[0] - color_b[0])
        split_interval = 180 - single_interval
        return InRangeHSV.hue_split \
            if split_interval < single_interval and interval_to_select == InRangeHSV.SMALLEST_HUE_INTERVAL \
               or split_interval > single_interval and interval_to_select == InRangeHSV.LARGEST_HUE_INTERVAL \
            else InRangeHSV.hue_single

    @staticmethod
    def hue_smallest(image, color_a, color_b):
        hue_method = InRangeHSV.choose_hue_method(color_a, color_b, InRangeHSV.SMALLEST_HUE_INTERVAL)
        return hue_method(image, color_a, color_b)

    @staticmethod
    def hue_largest(image, color_a, color_b):
        hue_method = InRangeHSV.choose_hue_method(color_a, color_b, InRangeHSV.LARGEST_HUE_INTERVAL)
        return hue_method(image, color_a, color_b)

    hue_modes_map = {
        "SMALLEST": hue_smallest,  # choose the smallest interval, independently of whether it requires a split or not
        "LARGEST": hue_largest,  # same as above but chooses the largest interval
        "IGNORE": hue_ignore,  # disregard hue entirely
        "SINGLE": hue_single,  # single check, ignores whether used interval is the smallest or the largest
        "SPLIT": hue_split,  # splits the check and ignores whether used interval is the smallest or the largest
    }
    hue_modes = list(hue_modes_map.keys())
    HUE_MODE_SINGLE = hue_modes[3]
    HUE_MODE_SPLIT = hue_modes[4]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "rgb_image": ("IMAGE",),
            "color_a": ("HSV_COLOR",),
            "color_b": ("HSV_COLOR",),
            "hue_mode": (s.hue_modes, {"default": s.hue_modes[0]})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "thresh"
    CATEGORY = "Bmad/CV/Thresholding"

    def thresh(self, rgb_image, color_a, color_b, hue_mode):
        image = tensor2opencv(rgb_image, 3)
        image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        thresh = self.hue_modes_map[hue_mode](image, color_a, color_b)
        thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
        thresh = opencv2tensor(thresh)
        return (thresh,)


class DistanceTransform:
    distance_types_map = {
        "DIST_L2": cv.DIST_L2,
        "DIST_L1": cv.DIST_L1,
        "DIST_C": cv.DIST_C,
    }
    distance_types = list(distance_types_map.keys())

    mask_sizes_map = {
        "DIST_MASK_3": cv.DIST_MASK_3,
        "DIST_MASK_5": cv.DIST_MASK_5,
        "DIST_MASK_PRECISE": cv.DIST_MASK_PRECISE
    }
    mask_sizes = list(mask_sizes_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "distance_type": (s.distance_types, {"default": s.distance_types[0]}),
                "mask_size": (s.mask_sizes, {"default": s.mask_sizes[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Bmad/CV/Thresholding"

    def apply(self, binary_image, distance_type, mask_size):
        binary_image = tensor2opencv(binary_image, 1)
        distance_transform = cv.distanceTransform(
            binary_image,
            self.distance_types_map[distance_type],
            self.mask_sizes_map[mask_size])
        distance_transform = opencv2tensor(distance_transform)
        return (distance_transform, )


# TODO maybe add GainDivision

# endregion


# region Morphological operations

class MorphologicOperation:
    # I did not want to make this node, but alas, I found no suit w/ the top/black hat operation
    # so might as well make a generic node w/ all the operations
    # just return as BW and implement convert node

    operation_map = {
        "MORPH_ERODE": cv.MORPH_ERODE,
        "MORPH_DILATE": cv.MORPH_DILATE,
        "MORPH_OPEN": cv.MORPH_OPEN,
        "MORPH_CLOSE": cv.MORPH_CLOSE,
        "MORPH_GRADIENT": cv.MORPH_GRADIENT,
        "MORPH_TOPHAT": cv.MORPH_TOPHAT,
        "MORPH_BLACKHAT": cv.MORPH_BLACKHAT,
    }
    operations = list(operation_map.keys())

    kernel_types_map = {
        "MORPH_RECT": cv.MORPH_RECT,
        "MORPH_ELLIPSE": cv.MORPH_ELLIPSE,
        "MORPH_CROSS": cv.MORPH_CROSS,
    }
    kernel_types = list(kernel_types_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",),
                "operation": (s.operations, s.operations),
                "kernel_type": (s.kernel_types, s.kernel_types),
                "kernel_size_x": ("INT", {"default": 4, "min": 2, "step": 2}),
                "kernel_size_y": ("INT", {"default": 4, "min": 2, "step": 2}),
                "iterations": ("INT", {"default": 1, "step": 1}),
            },
            # TODO maybe add optional input for custom kernel
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "Bmad/CV/Morphology"

    def apply(self, src, operation, kernel_type, kernel_size_x, kernel_size_y, iterations):
        img = tensor2opencv(src, 1)
        kernel = cv.getStructuringElement(self.kernel_types_map[kernel_type], (kernel_size_x + 1, kernel_size_y + 1))
        for i in range(iterations):
            img = cv.morphologyEx(img, self.operation_map[operation], kernel)
        return (opencv2tensor(img),)


class MorphologicSkeletoning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src": ("IMAGE",),
                #               "stop_criteria": (s.stop_criteria, {"default": s.stop_criteria[0]}),
                #               "iteration_limit": ("INT", {"default": 12, "min": -1, "step": 1}),
                # just use method,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = "Bmad/CV/Morphology"

    def compute(self, src):
        from skimage.morphology import skeletonize
        img = tensor2opencv(src)
        skel = skeletonize(img)
        img = opencv2tensor(skel)
        return (img,)


# endregion


# region color analysis

class ColorDefaultDictionary:
    default_color_dict = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "purple": (128, 0, 128),
        "teal": (0, 128, 128),
        "orange": (255, 165, 0),
        "pink": (255, 192, 203),
        #    "brown": (165, 42, 42),
        #    "gray": (128, 128, 128),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"number_of_colors": ("INT", {"default": 8, "min": 2, "max": 12})}}

    RETURN_TYPES = ("COLOR_DICT",)
    FUNCTION = "ret"
    CATEGORY = "Bmad/CV/Color A."

    def ret(self, number_of_colors):
        dic = dict(list(self.default_color_dict.items())[0: number_of_colors])
        return (dic,)


class FindComplementaryColor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "color_dict": ("COLOR_DICT",),
            "power": ("FLOAT", {"default": 2, "min": 2, "max": 10, "step": "0.2"}),
        },
            "optional":
                {
                    "mask": ("IMAGE",)
                }
        }

    RETURN_TYPES = ("COLOR", "STRING",)
    FUNCTION = "find_color"
    CATEGORY = "Bmad/CV/Color A."

    def find_color(self, image, color_dict, power, mask=None):
        image = tensor2opencv(image, 3)

        if mask is not None:
            mask = tensor2opencv(mask, 1)

            # this is a quality of life feature, so that it is easier to run the node and test stuff
            # the behavior (img resize w/ lin. interpolation) can be avoided by setting up the data prior to this node
            image = cv.resize(image, tuple(mask.shape), interpolation=cv.INTER_LINEAR)

        color = find_complementary_color(image, color_dict, mask, power)
        return (list(color_dict[color]), color,)


class SampleColorHSV:
    @classmethod
    def INPUT_TYPES(s):
        import sys
        return {"required": {
            "rgb_image": ("IMAGE",),
            "sample_size": ("INT", {"default": 1000, "min": 1, "max": 256 * 256, }),
            "sampling_seed": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1})
        }}

    RETURN_TYPES = ("HSV_SAMPLES", )
    FUNCTION = "sample"
    CATEGORY = "Bmad/CV/Color A."

    def sample(self, rgb_image, sample_size, sampling_seed):
        image = tensor2opencv(rgb_image, 3)
        image_width = image.shape[1]

        # sample pixels
        np.random.seed(sampling_seed)
        random_indices = np.random.choice(image.shape[0] * image_width, sample_size, replace=False)
        sample_pixels = np.array([image[i // image_width, i % image_width] for i in random_indices])
        sample_pixels = sample_pixels.reshape((1, -1, 3))

        # only convert samples to HSV
        sample_pixels_hsv = cv.cvtColor(sample_pixels, cv.COLOR_RGB2HSV)
        samples_object = HSV_Samples(sample_pixels_hsv[0, :, :])
        return (samples_object, )


class ColorToHSVColor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "rgb_color": ("COLOR",)
        }}

    RETURN_TYPES = ("HSV_COLOR",)
    FUNCTION = "convert"
    CATEGORY = "Bmad/CV/Color A."

    def convert(self, rgb_color):
        from colorsys import rgb_to_hsv
        rgb_color = setup_color_to_correct_type(rgb_color)
        (r, g, b) = tuple(rgb_color)
        rgb_color = (r / 255, g / 255, b / 255)
        (h, s, v) = rgb_to_hsv(*rgb_color)
        hsv = (int(h * 179), int(s * 255), int(v * 255))
        return (hsv,)


class KMeansColor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "number_of_colors": ("INT", {"default": 2, "min": 2}),
            "max_iterations": ("INT", {"default": 100}),
            "eps": ("FLOAT", {"default": .2, "step": 0.05})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_dominant_colors"
    CATEGORY = "Bmad/CV/Color A."

    def get_dominant_colors(self, image, number_of_colors, max_iterations, eps):
        image = tensor2opencv(image, 3)
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)

        # define criteria and apply kmeans
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iterations, eps)
        _, labels, centers = cv.kmeans(pixels, number_of_colors, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        # convert back into uint8, and make original image
        center = np.uint8(centers)
        res = center[labels.flatten()]
        res2 = res.reshape((image.shape))
        res2 = opencv2tensor(res2)
        return (res2,)


class BuildColorRangeHSV:
    @staticmethod
    def percentile(samples: HSV_Samples, percentage):
        value = percentage / 100 / 2
        bounds = {}
        bounds["h"] = samples.h_quant2(.5 - value, .5 + value)
        bounds["s"] = samples.s_quant2(.5 - value, .5 + value)
        bounds["v"] = samples.v_quant2(.5 - value, .5 + value)
        return bounds

    @staticmethod
    def avg_3maxdev(samples: HSV_Samples, percentage):
        value = percentage / 100
        bounds = {}
        bounds["h"] = [samples.h_avg - samples.h_max_dev * 3 * value, samples.h_avg + samples.h_max_dev * 3 * value]
        bounds["s"] = [samples.s_avg - samples.s_max_dev * 3 * value, samples.s_avg + samples.s_max_dev * 3 * value]
        bounds["v"] = [samples.v_avg - samples.v_max_dev * 3 * value, samples.v_avg + samples.v_max_dev * 3 * value]
        return bounds

    @staticmethod
    def avg_2stddev(samples: HSV_Samples, percentage):
        value = percentage / 100
        bounds = {}
        bounds["h"] = [samples.h_avg - samples.h_std_dev * 2 * value, samples.h_avg + samples.h_std_dev * 2 * value]
        bounds["s"] = [samples.s_avg - samples.s_std_dev * 2 * value, samples.s_avg + samples.s_std_dev * 2 * value]
        bounds["v"] = [samples.v_avg - samples.v_std_dev * 2 * value, samples.v_avg + samples.v_std_dev * 2 * value]
        return bounds

    @staticmethod
    def median_interpolate(samples: HSV_Samples, percentage):
        value = percentage / 100
        bounds = {}
        bounds["h"] = Interval([samples.h_median, samples.h_median]).interpolate(value, [0, 179])
        bounds["s"] = Interval([samples.s_median, samples.s_median]).interpolate(value, [0, 255])
        bounds["v"] = Interval([samples.v_median, samples.v_median]).interpolate(value, [0, 255])
        return bounds

    interval_modes_map = {
        "median to extremes interpolation": median_interpolate,
        "average +- 3x max deviation": avg_3maxdev,
        "average +- 2x standard deviation": avg_2stddev,
        "sample percentage centered at median": percentile,
    }
    interval_modes = list(interval_modes_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples": ("HSV_SAMPLES",),
            "percentage_modifier": ("INT", {"default": 50, "min": 1, "max": 100}),
            "interval_type": (s.interval_modes, s.interval_modes[0]),
        }}

    RETURN_TYPES = ("HSV_COLOR", "HSV_COLOR", InRangeHSV.hue_modes)
    FUNCTION = "get_interval"
    CATEGORY = "Bmad/CV/Color A."

    def get_interval(self, samples, percentage_modifier, interval_type):
        bounds = self.interval_modes_map[interval_type](samples, percentage_modifier)
        lower_bounds = np.array([bounds.get("h")[0], bounds.get("s")[0], bounds.get("v")[0]]).round()
        upper_bounds = np.array([bounds.get("h")[1], bounds.get("s")[1], bounds.get("v")[1]]).round()
        hue_mode = BuildColorRangeHSV.fix_bounds(lower_bounds, upper_bounds)
        return (upper_bounds, lower_bounds, hue_mode)

    @staticmethod
    def fix_bounds(lower_bounds, upper_bounds):
        # force hue bounds if interval >= 180
        interval_contains_zero = lower_bounds[0] <= 0  # example case: [-2, 2] includes the zero, but diff = 4
        if upper_bounds[0] - lower_bounds[0] >= (179 if interval_contains_zero else 180):
            lower_bounds[0] = 0
            upper_bounds[0] = 179  # note: return a color that exists, thus 179
        # check if hue needs to be split into 2 intervals when using inRange
        # note: 180 means zero is included, a one value split
        hue_mode = InRangeHSV.HUE_MODE_SPLIT \
            if lower_bounds[0] < 0 or upper_bounds[0] >= 180 \
            else InRangeHSV.HUE_MODE_SINGLE
        # correct hue bounds to [0, 180[
        lower_bounds[0] = (lower_bounds[0] + 180) % 180
        upper_bounds[0] = upper_bounds[0] % 180
        # clamp saturation and value limits to return actual colors in the outputs
        lower_bounds[1] = max(lower_bounds[1], 0)
        lower_bounds[2] = max(lower_bounds[2], 0)
        upper_bounds[1] = min(upper_bounds[1], 255)
        upper_bounds[2] = min(upper_bounds[2], 255)
        return hue_mode


class BuildColorRangeHSVAdvanced:
    def __init__(self):
        self.samples = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            # "average": ("HSV_COLOR",), # compute from sample?
            "samples": ("HSV_SAMPLES",),
            "hue_exp": ("STRING", {"multiline": True, "default": s.default_hue_expression}),
            "sat_exp": ("STRING", {"multiline": True, "default": s.default_saturation_expression}),
            "val_exp": ("STRING", {"multiline": True, "default": s.default_value_expression}),
        }}

    RETURN_TYPES = ("HSV_COLOR", "HSV_COLOR", InRangeHSV.hue_modes)
    FUNCTION = "get_interval"
    CATEGORY = "Bmad/CV/Color A."

    default_hue_expression = """# hue
h_quant2(0, 1).scale_by_constant(16) if 2 < v_median < 253 else to_interval(0, 180)
    """
    default_saturation_expression = """# saturation
to_interval(5, 255) if 2 < v_median < 253 else s_quant2(0,1).interpolate(0.2, [0, 255])
    """
    default_value_expression = """# value
v_quant2(0,1).interpolate(.5, [0, 255]).scale_by_constant(50) if 2 < v_median < 253 else v_quant2(0,1).scale_by_constant(8)
    """

    def get_interval(self, samples, hue_exp, sat_exp, val_exp):
        self.samples = samples

        # function to get sample names to include (avoids pre computing everything)
        # this supposes some computations could take considerable time, thus avoiding them if not used
        def valid_token(token: str):
            return token in samples_names and (
                    token.startswith("h_") or token.startswith("s_") or token.startswith("v_") or
                    token == "to_interval" or token == "minmax" or token == "maxmin"
            )

        # get bounds from the eval expressions
        bounds = {}
        samples_names = dir(samples)
        for key, expression in {"h": hue_exp, "s": sat_exp, "v": val_exp}.items():
            expression = prepare_text_for_eval(expression)  # purge potentially dangerous tokens

            locals_to_include_names = filter_expression_names(valid_token, expression)
            locals_to_include = {
                name: getattr(samples, name)
                for name in locals_to_include_names
            }

            bounds[key] = eval(expression, {
                "__builtins__": {},
                'min': min, 'max': max, 'm': math,
                **locals_to_include
            }, {})

        # get 2 colors that represent the computed lower and upper bounds
        lower_bounds = np.array([bounds.get("h")[0], bounds.get("s")[0], bounds.get("v")[0]]).round()
        upper_bounds = np.array([bounds.get("h")[1], bounds.get("s")[1], bounds.get("v")[1]]).round()
        hue_mode = BuildColorRangeHSV.fix_bounds(lower_bounds, upper_bounds)
        return (upper_bounds, lower_bounds, hue_mode)


# endregion


NODE_CLASS_MAPPINGS = {
    "ConvertImg": ConvertImg,
    "CopyMakeBorder": CopyMakeBorderSimple,
    "AddAlpha": AddAlpha,
    "FadeMaskEdges": FadeMaskEdges,

    "Framed Mask Grab Cut": FramedMaskGrabCut,
    "Framed Mask Grab Cut 2": FramedMaskGrabCut2,
    "Rect Grab Cut": RectGrabCut,

    "Contours": Contours,
    "Draw Contour(s)": DrawContours,
    "Get Contour from list": GetContourFromList,
    "BoundingRect (contours)": ContourGetBoundingRect,
    "Filter Contour": FilterContour,
    "Contour To Mask": ContourToMask,

    "SeamlessClone": SeamlessClone,
    "SeamlessClone (simple)": SeamlessCloneSimpler,
    "Inpaint": Inpaint,
    "ChameleonMask": ChameleonMask,

    "OtsuThreshold": OtsuThreshold,
    "AdaptiveThresholding": AdaptiveThresholding,
    "EqualizeHistogram": EqualizeHistogram,
    "CLAHE": CLAHE,
    "FindThreshold": FindThreshold,
    "InRange (hsv)": InRangeHSV,
    "DistanceTransform": DistanceTransform,
    # note: invert already exist: should be named ImageInvert, unless "overwritten" by some other custom node

    "MorphologicOperation": MorphologicOperation,
    "MorphologicSkeletoning": MorphologicSkeletoning,

    "ColorDictionary": ColorDefaultDictionary,
    "FindComplementaryColor": FindComplementaryColor,
    "KMeansColor": KMeansColor,
    "RGB to HSV": ColorToHSVColor,
    "SampleColorHSV": SampleColorHSV,
    "BuildColorRangeHSV (hsv)": BuildColorRangeHSV,
    "BuildColorRangeAdvanced (hsv)": BuildColorRangeHSVAdvanced,
}
