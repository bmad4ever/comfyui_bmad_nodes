from abc import ABC
import numpy as np
import cv2 as cv
import math
from .utils.dry import (tensor2opencv, opencv2tensor, image_output_formats_options, rect_modes, rect_modes_map,
                        maybe_convert_img, image_output_formats_options_map, prepare_text_for_eval, cache_with_ids,
                        filter_expression_names, base_category_path, images_category_path, print_yellow)
from .utils.color import (ImageColor, setup_color_to_correct_type, find_complementary_color, HSV_Samples, Interval)
from .utils.templates import ComboWrapperNode

# TODO these nodes return the mask, not the image with the background removed!
#       this is somewhat misleading. Consider changing the methods names.
#       ( but to what? GrabCutMask? FramedMaskGrabCutMask? ...)

# region types and constants

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
    'BORDER_DEFAULT': cv.BORDER_DEFAULT,
    'BORDER_ISOLATED': cv.BORDER_ISOLATED
}

border_types = list(border_types_map.keys())

border_types_excluding_transparent = border_types_map.copy()
border_types_excluding_transparent.pop("BORDER_TRANSPARENT")
border_types_excluding_transparent = list(border_types_excluding_transparent.keys())

interpolation_types_map = {
    "INTER_NEAREST": cv.INTER_NEAREST,
    "INTER_LINEAR": cv.INTER_LINEAR,
    "INTER_AREA": cv.INTER_AREA,
    "INTER_LANCZOS4": cv.INTER_LANCZOS4,
    "INTER_CUBIC": cv.INTER_CUBIC,
    #    "INTER_LINEAR_EXACT": cv.INTER_LINEAR_EXACT,
    #    "INTER_NEAREST_EXACT": cv.INTER_NEAREST_EXACT,
}
interpolation_types = list(interpolation_types_map.keys())

cv_category_path = f"{base_category_path}/CV"

# endregion


# region misc

class CopyMakeBorderSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "border_size": ("INT", {"default": 64}),
                "border_type": (border_types_excluding_transparent, {"default": border_types[0]})
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_border"
    CATEGORY = f"{cv_category_path}/Misc"

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
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "to": (cls.options, {"default": cls.options[1]})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = f"{cv_category_path}"

    def convert(self, image, to):
        image = tensor2opencv(image, self.options_map[to])
        return (opencv2tensor(image),)


class AddAlpha:
    method = ["default", "invert"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
            },
            "optional": {
                "alpha": ("IMAGE",),
                "method": (cls.method, {"default": cls.method[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_alpha"
    CATEGORY = images_category_path

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
    def INPUT_TYPES(cls):
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
    CATEGORY = f"{cv_category_path}/Misc"

    def apply(self, binary_image, edge_size, edge_tightness, edge_exponent, smoothing_diameter, paste_original_blacks):
        binary_image = tensor2opencv(binary_image, 1)
        # _, binary_image = cv.threshold(gray_image, 128, 255, cv.THRESH_BINARY) # suppose it's already binary

        # compute L2 (euclidean) distance -> normalize with respect to edge size -> smooth
        distance_transform = cv.distanceTransform(binary_image, cv.DIST_L2, cv.DIST_MASK_3)
        normalized_distance = distance_transform / edge_size
        smoothed_distance = cv.bilateralFilter(normalized_distance, smoothing_diameter, 75, 75)

        # darken the white pixels based on smoothed distance and "edge tightness"
        diff = 1 - smoothed_distance
        darkened_image = (abs(diff * edge_tightness) ** (1 / edge_exponent)) * np.sign(diff)
        darkened_image = np.clip(darkened_image, 0, 1)
        darkened_image = (darkened_image * 255).astype(np.uint8)

        if paste_original_blacks:  # mask original black pixels
            black_mask = binary_image < 1
            darkened_image[black_mask] = 0

        output_image = binary_image - darkened_image  # darken original image
        output_image = opencv2tensor(output_image)
        return (output_image,)


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
    def INPUT_TYPES(cls):
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
                "frame_option": (cls.frame_options, {
                    "default": cls.frame_options[0]
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
    CATEGORY = f"{cv_category_path}/GrabCut"

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
    def INPUT_TYPES(cls):
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

    CATEGORY = f"{cv_category_path}/GrabCut"

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
    def INPUT_TYPES(cls):
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
                "frame_option": (cls.frame_options, {
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

    CATEGORY = f"{cv_category_path}/GrabCut"

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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "retrieval_mode": (cls.retrieval_modes, {"default": "RETR_LIST"}),
                "approximation_mode": (cls.approximation_modes, {"default": "CHAIN_APPROX_SIMPLE"}),
            },
        }

    RETURN_TYPES = ("CV_CONTOURS", "CV_CONTOUR", "CV_CONTOURS_HIERARCHY")
    FUNCTION = "find_contours"
    CATEGORY = f"{cv_category_path}/Contour"
    OUTPUT_IS_LIST = (False, True, False)

    def find_contours(self, image, retrieval_mode, approximation_mode):
        image = tensor2opencv(image)
        thresh = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        # no thresh applied here, non zeroes are treated as 1 according to documentation;
        # thresh should have been already applied to the image, before passing it to this node.

        contours, hierarchy = cv.findContours(
            thresh,
            self.retrieval_modes_map[retrieval_mode],
            self.approximation_modes_map[approximation_mode])

        return (contours, contours, hierarchy,)


class DrawContours:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "contours": ("CV_CONTOURS",),
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

    CATEGORY = f"{cv_category_path}/Contour"

    def draw(self, image, contours, index_to_draw, color, thickness):
        background = tensor2opencv(image)

        um_image = cv.UMat(background)
        cv.drawContours(um_image, contours, index_to_draw, ImageColor.getcolor(color, "RGB"), thickness)
        contour_image = um_image.get()

        image = opencv2tensor(contour_image)

        return (image,)


class GetContourFromList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "contours": ("CV_CONTOURS",),
                "index": ("INT", {"default": 0, "min": 0, "step": 1})
            }
        }

    RETURN_TYPES = ("CV_CONTOUR",)
    FUNCTION = "get_contour"
    CATEGORY = f"{cv_category_path}/Contour"

    def get_contour(self, contours, index):
        if index >= len(contours):
            return (None,)
        return (contours[index],)


class ContourGetBoundingRect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "contour": ("CV_CONTOUR",),
                "return_mode": (rect_modes, {"default": rect_modes[1]})
            },
        }

    RETURN_TYPES = tuple(["INT" for _ in range(4)])
    FUNCTION = "compute"
    CATEGORY = f"{cv_category_path}/Contour"

    def compute(self, contour, return_mode):
        if contour is None:
            print("Contour = None !")
            return (0, 0, 0, 0,)

        # convert opencv boundingRect format to bounds
        bounds = rect_modes_map[rect_modes[0]]["toBounds"](*cv.boundingRect(contour))

        # convert from bounds to desired output format on return
        return rect_modes_map[return_mode]["fromBounds"](*bounds)


class FilterContour:
    @staticmethod
    def MODE(cnts, fit):
        sorted_list = sorted(cnts, key=fit)
        return [sorted_list[len(sorted_list) // 2]]

    return_modes_map = {
        "MAX": lambda cnts, fit: [sorted(cnts, key=fit)[-1]],
        "MIN": lambda cnts, fit: [sorted(cnts, key=fit)[0]],
        "MODE": MODE,
        "FILTER": lambda cnts, fit: list(filter(fit, cnts)),
    }
    return_modes = list(return_modes_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "contours": ("CV_CONTOURS",),
                "fitness": ("STRING", {"multiline": True, "default":
                    "# Contour Fitness Function\n"}),
                "select": (cls.return_modes, {"default": cls.return_modes[0]})
            },
            "optional": {
                "image": ("IMAGE",),
                "aux_contour": ("CV_CONTOUR",)
            }
        }

    RETURN_TYPES = ("CV_CONTOUR", "CV_CONTOURS")
    FUNCTION = "filter"
    CATEGORY = f"{cv_category_path}/Contour"

    def filter(self, contours, fitness, select, image=None, aux_contour=None):
        import math
        import cv2
        import numpy

        if len(contours) == 0:
            print("Contour list is empty")
            return ([[]], contours)

        # region prepare inputs
        if image is not None:
            image = tensor2opencv(image)

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
            _, _, w, h = boundingRect(cnt)
            return float(w) / h

        @cache_with_ids(single=True)
        def extent(cnt):
            area = contourArea(cnt)
            _, _, w, h = boundingRect(cnt)
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
            m = cv.moments(cnt)
            c_x = int(m["m10"] / m["m00"])
            c_y = int(m["m01"] / m["m00"])
            return c_x, c_y

        @cache_with_ids(single=False)
        def contour_mask(cnt, img):
            if len(img.shape) > 2:
                height, width, _ = img.shape
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
            intersection = cv2.bitwise_and(
                gray, cv2.drawContours(np.zeros_like(gray), [cnt], 0, 255, thickness=cv2.FILLED))
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

        ret = self.return_modes_map[select](contours, lambda c: fitness(c, image, aux_contour))
        return (ret[0], ret,)


class ContourToMask:
    @classmethod
    def INPUT_TYPES(cls):
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
    CATEGORY = f"{cv_category_path}/Contour"

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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dst": ("IMAGE",),
                "src": ("IMAGE",),
                "src_mask": ("IMAGE",),
                "flag": (cls.clone_modes, {"default": cls.clone_modes[0]}),
                "cx": ("INT", {"default": 0, "min": -999999, "step": 1}),
                "cy": ("INT", {"default": 0, "min": -999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"
    CATEGORY = f"{cv_category_path}/C.Photography"

    def paste(self, src, dst, src_mask, flag, cx, cy):
        src = tensor2opencv(src)
        dst = tensor2opencv(dst)
        src_mask = tensor2opencv(src_mask, 1)

        result = cv.seamlessClone(src, dst, src_mask, (cx, cy), self.clone_modes_map[flag])
        result = opencv2tensor(result)

        return (result,)


class SeamlessCloneSimpler:
    @classmethod
    def INPUT_TYPES(cls):
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
    CATEGORY = f"{cv_category_path}/C.Photography"

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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE",),
                "mask": ("IMAGE",),
                "radius": ("INT", {"default": 3, "min": 0, "step": 1}),
                "flag": (cls.inpaint_methods, {"default": cls.inpaint_methods[0]}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paint"
    CATEGORY = f"{cv_category_path}/C.Photography"

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
    def INPUT_TYPES(cls):
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
                "mode": (cls.modes, {"default": cls.modes[0]}),
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
    CATEGORY = f"{cv_category_path}/C.Photography"

    def create_mask(self, src, dst, thresh_blur, close_dist, open_dist, size_dist, mask_blur,
                    contrast_adjust, mode: str, output_format, optional_roi_mask=None):
        src = tensor2opencv(src)
        dst = tensor2opencv(dst)

        thresh_blur += 1
        if mask_blur > 0:
            mask_blur += 1

        # compute the difference between images based on mode
        src = self.mode_func_map[mode](src)  # type:ignore
        dst = self.mode_func_map[mode](dst)  # type:ignore
        diff = cv.absdiff(src, dst)

        if mode == "HUE":
            diff = np.minimum(diff, 180 - diff)

        # binary thresholding
        # _, mask = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
        diff = cv.GaussianBlur(diff, (thresh_blur, thresh_blur), 0)
        _, mask = cv.threshold(diff, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
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
    def INPUT_TYPES(cls):
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
    CATEGORY = f"{cv_category_path}/Thresholding"

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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                "max_value": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                # maybe should just allow for 255? may just confuse some people that don't read documentation
                "adaptive_method": (cls.adaptive_modes, {"default": cls.adaptive_modes[1]}),
                "threshold_type": (thresh_types, {"default": thresh_types[0]}),
                "block_size": ("INT", {"default": 4, "min": 2, "step": 2}),
                "c": ("INT", {"default": 2, "min": -999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "thresh"
    CATEGORY = f"{cv_category_path}/Thresholding"

    def thresh(self, src, max_value, adaptive_method, threshold_type, block_size, c):
        # maybe allow to use from a specific channel 1st? nah, just create a node to fetch the channel
        # might be useful for other nodes
        src = tensor2opencv(src, 1)
        src = cv.adaptiveThreshold(src, max_value, self.adaptive_modes_map[adaptive_method],
                                   thresh_types_map[threshold_type], block_size + 1, c)
        src = cv.cvtColor(src, cv.COLOR_GRAY2RGB)
        src = opencv2tensor(src)
        return (src,)


class EqualizeHistogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "eq"
    CATEGORY = f"{cv_category_path}/Thresholding"

    def eq(self, src):
        src = tensor2opencv(src, 1)
        eq = cv.equalizeHist(src)
        eq = cv.cvtColor(eq, cv.COLOR_GRAY2RGB)
        eq = opencv2tensor(eq)
        return (eq,)


class CLAHE:
    @classmethod
    def INPUT_TYPES(cls):
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
    CATEGORY = f"{cv_category_path}/Thresholding"

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
    def INPUT_TYPES(cls):
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
    CATEGORY = f"{cv_category_path}/Thresholding"

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


#class IN_RANGE_HUE_MODE:
#    @classmethod
#    def INPUT_TYPES(cls):
#        return {"required": {
#            "hue_mode": (InRangeHSV.hue_modes, {"default": InRangeHSV.hue_modes[0]})
#        }}
#
#    RETURN_TYPES = ("IR_HUE_MODE",)
#    FUNCTION = "ret"
#    CATEGORY = f"{cv_category_path}/Thresholding"
#
#    def ret(self, hue_mode):
#        return (hue_mode,)

class InRangeHSV:
    # w/ respect to documentation in :
    #   https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
    # both bounds are inclusive

    # Hardcoded for 180 degrees hue

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
    def INPUT_TYPES(cls):
        return {"required": {
            "rgb_image": ("IMAGE",),
            "color_a": ("HSV_COLOR",),
            "color_b": ("HSV_COLOR",),
            "hue_mode": (IN_RANGE_HUE_MODE.TYPE_NAME,)
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "thresh"
    CATEGORY = f"{cv_category_path}/Thresholding"

    def thresh(self, rgb_image, color_a, color_b, hue_mode):
        image = tensor2opencv(rgb_image, 3)
        image = cv.cvtColor(image, cv.COLOR_RGB2HSV)
        thresh = self.hue_modes_map[hue_mode](image, color_a, color_b)
        thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
        thresh = opencv2tensor(thresh)
        return (thresh,)


class IN_RANGE_HUE_MODE(metaclass=ComboWrapperNode):
    TYPE_NAME = "IR_HUE_MODE"
    OPTIONS_LIST = InRangeHSV.hue_modes
    CATEGORY = f"{cv_category_path}/Thresholding"


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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "binary_image": ("IMAGE",),
                "distance_type": (cls.distance_types, {"default": cls.distance_types[0]}),
                "mask_size": (cls.mask_sizes, {"default": cls.mask_sizes[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = f"{cv_category_path}/Thresholding"

    def apply(self, binary_image, distance_type, mask_size):
        binary_image = tensor2opencv(binary_image, 1)
        distance_transform = cv.distanceTransform(
            binary_image,
            self.distance_types_map[distance_type],
            self.mask_sizes_map[mask_size])
        distance_transform = opencv2tensor(distance_transform)
        return (distance_transform,)


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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                "operation": (cls.operations, {"default": cls.operations[0]}),
                "kernel_type": (cls.kernel_types, {"default": cls.kernel_types[0]}),
                "kernel_size_x": ("INT", {"default": 4, "min": 2, "step": 2}),
                "kernel_size_y": ("INT", {"default": 4, "min": 2, "step": 2}),
                "iterations": ("INT", {"default": 1, "step": 1}),
            },
            # TODO maybe add optional input for custom kernel
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = f"{cv_category_path}/Morphology"

    def apply(self, src, operation, kernel_type, kernel_size_x, kernel_size_y, iterations):
        img = tensor2opencv(src, 1)
        kernel = cv.getStructuringElement(self.kernel_types_map[kernel_type], (kernel_size_x + 1, kernel_size_y + 1))
        for i in range(iterations):
            img = cv.morphologyEx(img, self.operation_map[operation], kernel)
        return (opencv2tensor(img),)


class MorphologicSkeletoning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = f"{cv_category_path}/Morphology"

    def compute(self, src):
        from skimage.morphology import skeletonize
        img = tensor2opencv(src, 1)
        _, img = cv.threshold(img, 127, 1, cv.THRESH_BINARY)  # ensure it is binary and set max value to 1.
        skel = skeletonize(img) * 255
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
    def INPUT_TYPES(cls):
        return {"required": {"number_of_colors": ("INT", {"default": 8, "min": 2, "max": 12})}}

    RETURN_TYPES = ("COLOR_DICT",)
    FUNCTION = "ret"
    CATEGORY = f"{cv_category_path}/Color A."

    def ret(self, number_of_colors):
        dic = dict(list(self.default_color_dict.items())[0: number_of_colors])
        return (dic,)


class ColorCustomDictionary:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "color_names": ("STRING", {"default": ""}),
            "colors": ("COLOR", {"default": ""})
        }
        }

    RETURN_TYPES = ("COLOR_DICT",)
    FUNCTION = "ret"
    CATEGORY = f"{cv_category_path}/Color A."
    INPUT_IS_LIST = True

    def ret(self, color_names, colors):
        if len(color_names) != len(colors):
            print_yellow("color_names size is different than colors size!")
            min_len = min(len(color_names), len(colors))
            color_names = color_names[0:min_len]
            colors = colors[0:min_len]

        return (dict(zip(color_names, colors)),)


class FindComplementaryColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "color_dict": ("COLOR_DICT",),
            "power": ("FLOAT", {"default": 0.5, "min": .01, "max": 10, "step": 0.01}),
        },
            "optional":
                {
                    "mask": ("IMAGE",)
                }
        }

    RETURN_TYPES = ("COLOR", "STRING",)
    FUNCTION = "find_color"
    CATEGORY = f"{cv_category_path}/Color A."

    def find_color(self, image, color_dict, power, mask=None):
        image = tensor2opencv(image, 3)

        if mask is not None:
            mask = tensor2opencv(mask, 1)

            # this is a quality of life feature, so that it is easier to run the node and test stuff
            # the behavior (img resize w/ lin. interpolation) can be avoided by setting up the data prior to this node
            if image.shape[0:2] != mask.shape[0:2]:
                print("FindComplementaryColor node will resize image to fit mask.")
                image = cv.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv.INTER_LINEAR)

        color = find_complementary_color(image, color_dict, mask, power)
        return (list(color_dict[color]), color,)


class SampleColorHSV:
    @classmethod
    def INPUT_TYPES(cls):
        import sys
        return {"required": {
            "rgb_image": ("IMAGE",),
            "sample_size": ("INT", {"default": 1000, "min": 1, "max": 256 * 256, }),
            "sampling_seed": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1})
        }}

    RETURN_TYPES = ("HSV_SAMPLES",)
    FUNCTION = "sample"
    CATEGORY = f"{cv_category_path}/Color A."

    def sample(self, rgb_image, sample_size, sampling_seed):
        image = tensor2opencv(rgb_image, 3)
        image_width = image.shape[1]

        # sample pixels
        rng = np.random.default_rng(seed=sampling_seed)
        random_indices = rng.choice(image.shape[0] * image_width, sample_size, replace=False)
        sample_pixels = np.array([image[i // image_width, i % image_width] for i in random_indices])
        sample_pixels = sample_pixels.reshape((1, -1, 3))

        # only convert samples to HSV
        sample_pixels_hsv = cv.cvtColor(sample_pixels, cv.COLOR_RGB2HSV)
        samples_object = HSV_Samples(sample_pixels_hsv[0, :, :])
        return (samples_object,)


class ColorToHSVColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rgb_color": ("COLOR",)
        }}

    RETURN_TYPES = ("HSV_COLOR",)
    FUNCTION = "convert"
    CATEGORY = f"{cv_category_path}/Color A."

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
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "number_of_colors": ("INT", {"default": 2, "min": 1}),
            "max_iterations": ("INT", {"default": 100}),
            "eps": ("FLOAT", {"default": .2, "step": 0.05})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_colors"
    CATEGORY = f"{cv_category_path}/Color A."

    def get_colors(self, image, number_of_colors, max_iterations, eps):
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


class NaiveAutoKMeansColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "max_k": ("INT", {"default": 8, "min": 3, "max": 16}),

            # besides looking like the elbow,
            #  a k's compactness divided the by first computed compactness should be below this value
            "rc_threshold": ("FLOAT", {"default": .5, "max": 1, "min": 0.01, "step": 0.01}),

            "max_iterations": ("INT", {"default": 100}),
            "eps": ("FLOAT", {"default": .2, "step": 0.05})
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    FUNCTION = "get_colors"
    CATEGORY = f"{cv_category_path}/Color A."

    def get_colors(self, image, max_k, rc_threshold, max_iterations, eps):
        image = tensor2opencv(image, 3)
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)

        def normalize(vector):
            return vector / np.linalg.norm(vector)

        def compute_angle_at_k(prev_k_c, k_c, next_k_c):
            p_km1 = np.array([-1, prev_k_c, 0])
            p_k = np.array([0, k_c, 0])
            p_kp1 = np.array([1, next_k_c, 0])
            v1 = normalize(p_km1 - p_k)
            v2 = normalize(p_kp1 - p_k)
            return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

        # define criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iterations, eps)

        # compute k means and check for the elbow
        #  here the elbow is the edgiest point on the compactness graph
        best_angle = 7  # max is pi, when the line is perfectly straight; and the objective is to minimize the angle
        max_c = best_rc = best_k = None
        current_k_data = prev_k_data = best_k_data = None
        for k in range(1, max_k + 2):
            next_k_data = cv.kmeans(pixels, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

            if max_c is None:
                max_c = next_k_data[0]

            if next_k_data[0] == 0:
                # if it is a perfect fit, leave the method
                # avoids unneeded computation, and division by zero on k = 1
                best_k_data = next_k_data
                best_k = k
                break

            if k > 2:
                rc = current_k_data[0] / max_c
                angle = compute_angle_at_k(prev_k_data[0] / max_c, rc, next_k_data[0] / max_c)
                if angle < best_angle or best_rc > rc_threshold:
                    best_angle = angle
                    best_k_data = current_k_data
                    best_rc = rc
                    best_k = k - 1

            prev_k_data = current_k_data
            current_k_data = next_k_data

        # convert back into uint8, and make original image
        center = np.uint8(best_k_data[2])
        res = center[best_k_data[1].flatten()]
        res2 = res.reshape((image.shape))
        res2 = opencv2tensor(res2)
        return (res2, best_k)


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
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("HSV_SAMPLES",),
            "percentage_modifier": ("INT", {"default": 50, "min": 1, "max": 100}),
            "interval_type": (cls.interval_modes, {"default": cls.interval_modes[0]}),
        }}

    RETURN_TYPES = ("HSV_COLOR", "HSV_COLOR", IN_RANGE_HUE_MODE.TYPE_NAME)
    FUNCTION = "get_interval"
    CATEGORY = f"{cv_category_path}/Color A."

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
    def INPUT_TYPES(cls):
        return {"required": {
            # "average": ("HSV_COLOR",), # compute from sample?
            "samples": ("HSV_SAMPLES",),
            "hue_exp": ("STRING", {"multiline": True, "default": cls.default_hue_expression}),
            "sat_exp": ("STRING", {"multiline": True, "default": cls.default_saturation_expression}),
            "val_exp": ("STRING", {"multiline": True, "default": cls.default_value_expression}),
        }}

    RETURN_TYPES = ("HSV_COLOR", "HSV_COLOR", IN_RANGE_HUE_MODE.TYPE_NAME)
    FUNCTION = "get_interval"
    CATEGORY = f"{cv_category_path}/Color A."

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


# region transforms


class Remap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "remap": ("REMAP", {"forceInput": True}),
            "src": ("IMAGE",),
            "interpolation": (interpolation_types, {"default": interpolation_types[2]}),
        },
            "optional": {
                "src_mask": ("MASK",),
                "output_with_alpha": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "transform"
    CATEGORY = f"{cv_category_path}/Transform"

    def transform(self, src, remap, interpolation, src_mask=None, output_with_alpha=False):
        src = tensor2opencv(src)
        dst_dims = remap["dims"] if "dims" in remap else src.shape[:2]
        func = remap["func"]
        xargs = remap["xargs"]
        # if src_mask is not defined set it to a blank canvas; otherwise, just unwrap it
        src_mask = np.ones(src.shape[:2]) * 255 if src_mask is None else tensor2opencv(src_mask, 1)

        if "custom" not in remap.keys():
            # generic application, using cv.remap
            xs, ys, bb = func(src, *xargs)
            remap_img = cv.remap(src, xs, ys, interpolation_types_map[interpolation])
            mask = cv.remap(src_mask, xs, ys, interpolation_types_map[interpolation])
        else:
            # non-generic application; replaces cv.remap w/ some other function.
            # so far only for user provided homography,
            #  to avoid a separate node, since technically it is also a remap and also uses the interpolation argument.
            custom_data = func(src, *xargs)
            remap_img, mask, bb = remap["custom"](custom_data, src, interpolation_types_map[interpolation], src_mask)

        if bb is not None:
            new_img = np.zeros((*dst_dims, 3))  # hope width and height are not swapped
            new_img[bb[1]:bb[3], bb[0]:bb[2], :] = remap_img
            remap_img = new_img
            new_img = np.zeros(dst_dims)  # was working previously without the batch dim; unsure if really needed
            new_img[bb[1]:bb[3], bb[0]:bb[2]] = mask
            mask = new_img

        if output_with_alpha:
            new_img = np.zeros((*dst_dims, 4))
            new_img[:, :, 0:3] = remap_img[:, :, :]
            new_img[:, :, 3] = mask[:, :]
            remap_img = new_img

        return (opencv2tensor(remap_img), opencv2tensor(mask))


class RemapBase(ABC):
    RETURN_TYPES = ("REMAP",)
    FUNCTION = "send_remap"
    CATEGORY = f"{cv_category_path}/Transform"

    @staticmethod
    def get_dims(mask):
        _, h, w = mask.shape
        return h, w


class InnerCylinderRemap(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fov": ("INT", {"default": 90, "min": 1, "max": 179}),
            "swap_xy": ("BOOLEAN", {"default": False}),
        }
        }

    def send_remap(self, fov, swap_xy):
        from .utils.remaps import remap_inner_cylinder
        return ({
                    "func": remap_inner_cylinder,
                    "xargs": [fov, swap_xy]
                },)


class OuterCylinderRemap(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fov": ("INT", {"default": 90, "min": 1, "max": 179}),
            "swap_xy": ("BOOLEAN", {"default": False}),
        }
        }

    def send_remap(self, fov, swap_xy):
        from .utils.remaps import remap_outer_cylinder
        return ({
                    "func": remap_outer_cylinder,
                    "xargs": [fov, swap_xy]
                },)


class RemapPinch(RemapBase):
    INPUT_TYPES_DICT = {
        "required": {
            "power_x": ("FLOAT", {"default": 1, "min": 1, "max": 3, "step": .05}),
            "power_y": ("FLOAT", {"default": 1, "min": 1, "max": 3, "step": .05}),
            "center_x": ("FLOAT", {"default": .5, "min": 0, "max": 1, "step": .05}),
            "center_y": ("FLOAT", {"default": .5, "min": 0, "max": 1, "step": .05}),
        }
    }

    @classmethod
    def INPUT_TYPES(cls):
        return RemapPinch.INPUT_TYPES_DICT

    def send_remap(self, power_x, power_y, center_x, center_y):
        from .utils.remaps import remap_pinch_or_stretch
        return ({
                    "func": remap_pinch_or_stretch,
                    "xargs": [(power_x, power_y), (center_x, center_y)]
                },)


class RemapStretch(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return RemapPinch.INPUT_TYPES_DICT

    def send_remap(self, power_x, power_y, center_x, center_y):
        from .utils.remaps import remap_pinch_or_stretch
        return ({
                    "func": remap_pinch_or_stretch,
                    "xargs": [(1 / power_x, 1 / power_y), (center_x, center_y)]
                },)


class RemapBarrelDistortion(RemapBase):
    @staticmethod
    def BARREL_DIST_TYPES():
        return {
            "required":
                {
                    "a": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.00001}),
                    "b": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.00001}),
                    "c": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 0.00001}),
                    "use_inverse_variant": ("BOOLEAN", {"default": True})
                },
            "optional": {
                "d": ("FLOAT", {"forceInput": True})
            }
        }

    @classmethod
    def INPUT_TYPES(cls):
        return RemapBarrelDistortion.BARREL_DIST_TYPES()
        # inputs = RemapBarrelDistortion.BARREL_DIST_F_TYPES()
        # inputs["required"]["use_inverse_variant"] = ("BOOLEAN", {"default": True})
        # return inputs

    def send_remap(self, a, b, c, use_inverse_variant, d=None):
        from .utils.remaps import remap_barrel_distortion
        return ({
                    "func": remap_barrel_distortion,
                    "xargs": [a, b, c, d, use_inverse_variant]
                },)


class RemapReverseBarrelDistortion(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return RemapBarrelDistortion.BARREL_DIST_TYPES()

    def send_remap(self, a, b, c, use_inverse_variant, d=None):
        from .utils.remaps import remap_reverse_barrel_distortion
        return ({
                    "func": remap_reverse_barrel_distortion,
                    "xargs": [a, b, c, d, use_inverse_variant]
                },)


class RemapInsideParabolas(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dst_mask_with_2_parabolas": ("MASK",),
        }
        }

    def send_remap(self, dst_mask_with_2_parabolas):
        from .utils.remaps import remap_inside_parabolas_simple
        return ({
                    "func": remap_inside_parabolas_simple,
                    "xargs": [tensor2opencv(dst_mask_with_2_parabolas, 1)],
                    "dims": RemapBase.get_dims(dst_mask_with_2_parabolas)
                },)


class RemapInsideParabolasAdvanced(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dst_mask_with_2_parabolas": ("MASK",),
            "curve_wise_adjust": ("FLOAT", {"default": 1, "min": .3, "max": 2, "step": .01}),
            "ortho_wise_adjust": ("FLOAT", {"default": 1, "min": 1, "max": 3, "step": .01}),
            "flip_ortho": ("BOOLEAN", {"default": False})
        }
        }

    def send_remap(self, dst_mask_with_2_parabolas, curve_wise_adjust, ortho_wise_adjust, flip_ortho):
        from .utils.remaps import remap_inside_parabolas_advanced
        return ({
                    "func": remap_inside_parabolas_advanced,
                    "xargs": [tensor2opencv(dst_mask_with_2_parabolas, 1),
                              curve_wise_adjust, ortho_wise_adjust, flip_ortho],
                    "dims": RemapBase.get_dims(dst_mask_with_2_parabolas)
                },)


class RemapFromInsideParabolas(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "src_mask_with_2_parabolas": ("MASK",),
            "width": ("INT", {"default": 512, "min": 16, "max": 4096}),
            "height": ("INT", {"default": 512, "min": 16, "max": 4096}),
        }
        }

    def send_remap(self, src_mask_with_2_parabolas, width, height):
        from .utils.remaps import remap_from_inside_parabolas
        return ({
                    "func": remap_from_inside_parabolas,
                    "xargs": [tensor2opencv(src_mask_with_2_parabolas, 1), width, height],
                    "dims": (width, height)
                },)


class RemapQuadrilateral(RemapBase):
    from .utils.remaps import quad_remap_methods_map

    modes_list = list(quad_remap_methods_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dst_mask_with_4_points": ("MASK",),
            "mode": (cls.modes_list, {"default": cls.modes_list[0]}),
        }
        }

    @staticmethod
    def homography(custom_data, src, interpolation, mask=None):
        h_matrix, bb = custom_data
        bb_width, bb_height = bb[2] - bb[0], bb[3] - bb[1]
        ret = cv.warpPerspective(src, h_matrix, (bb_width, bb_height), flags=interpolation,
                                 borderMode=cv.BORDER_CONSTANT)
        if mask is not None:
            mask = cv.warpPerspective(mask, h_matrix, (bb_width, bb_height), flags=interpolation,
                                      borderMode=cv.BORDER_CONSTANT)
        return ret, mask, bb

    def send_remap(self, dst_mask_with_4_points, mode):
        from .utils.remaps import remap_quadrilateral
        remap_data = {
            "func": remap_quadrilateral,
            "xargs": [tensor2opencv(dst_mask_with_4_points, 1), mode],
            "dims": RemapBase.get_dims(dst_mask_with_4_points)
        }
        if mode == "HOMOGRAPHY":
            remap_data["custom"] = RemapQuadrilateral.homography
        return (remap_data,)


class RemapFromQuadrilateral(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "src_mask_with_4_points": ("MASK",),
            # "mode": (s.modes_list, {"default": s.modes_list[0]}),
            "width": ("INT", {"default": 512, "min": 16, "max": 4096}),
            "height": ("INT", {"default": 512, "min": 16, "max": 4096}),
        }
        }

    @staticmethod
    def homography(*args):
        ret, mask, bb = RemapQuadrilateral.homography(*args)
        return ret, mask, None

    def send_remap(self, src_mask_with_4_points, width, height):
        from .utils.remaps import remap_from_quadrilateral
        remap_data = {
            "func": remap_from_quadrilateral,
            "xargs": [tensor2opencv(src_mask_with_4_points, 1), width, height],
            "dims": (width, height),  # seems kinda redundant, not sure if should refactor
            "custom": RemapFromQuadrilateral.homography
        }
        return (remap_data,)


class RemapWarpPolar(RemapBase):
    MAX_RADIUS = {
        "half min shape": lambda shape: np.min(shape[:2]) / 2,
        "half max shape": lambda shape: np.max(shape[:2]) / 2,
        "hypot": lambda shape: np.hypot(shape[1] / 2, shape[0] / 2),
        "raw": lambda _: 1  # uses value set by radius_adjust
    }
    MAX_RADIUS_KEYS = list(MAX_RADIUS.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "max_radius": (cls.MAX_RADIUS_KEYS, {"default": cls.MAX_RADIUS_KEYS[0]}),
            "radius_adjust": ("FLOAT", {"default": 1, "min": .1, "max": 2048, "step": 0.01}),
            "center_x_adjust": ("FLOAT", {"default": 0, "min": -3, "max": 3, "step": 0.01}),
            "center_y_adjust": ("FLOAT", {"default": 0, "min": -3, "max": 3, "step": 0.01}),
            "log": ("BOOLEAN", {"default": False}),
            "inverse": ("BOOLEAN", {"default": True}),
            "crop": ("BOOLEAN", {"default": True})
        }
        }

    @staticmethod
    def warp(custom_data, src, interpolation, mask=None):
        max_radius, radius_adj, center_x_adj, center_y_adj, log, inverse, crop = custom_data

        center = (
        src.shape[1] / 2 + src.shape[1] / 2 * center_x_adj, src.shape[0] / 2 + src.shape[0] / 2 * center_y_adj)
        radius = RemapWarpPolar.MAX_RADIUS[max_radius](src.shape) * radius_adj
        flags = interpolation | cv.WARP_FILL_OUTLIERS
        flags |= cv.WARP_POLAR_LOG if log else cv.WARP_POLAR_LINEAR
        if inverse:
            flags |= cv.WARP_INVERSE_MAP

        img = cv.warpPolar(src, (src.shape[1], src.shape[0]), center, radius, flags)
        if mask is not None:
            mask = cv.warpPolar(mask, (mask.shape[1], mask.shape[0]), center, radius, flags)

        if crop:
            left, right = int(max(center[0] - radius, 0)), int(min(center[0] + radius, src.shape[1]))
            top, bottom = int(max(center[1] - radius, 0)), int(min(center[1] + radius, src.shape[0]))
            img = img[top:bottom, left:right]
            mask = mask[top:bottom, left:right]

        return img, mask, None

    def send_remap(self, max_radius, radius_adjust, center_x_adjust, center_y_adjust, log, inverse, crop):
        remap_data = {
            "func": lambda _, mr, ra, cx, cy, l, i, c: (mr, ra, cx, cy, l, i, c),  # does nothing, just returns args
            "xargs": [max_radius, radius_adjust, center_x_adjust, center_y_adjust, log, inverse, crop],
            "custom": RemapWarpPolar.warp
        }
        return (remap_data,)


# endregion


# region misc. ADVANCED


class MaskOuterBlur:  # great, another "funny" name; what would you call this?
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src": ("IMAGE",),
                "mask": ("IMAGE",),
                "kernel_size": ("INT", {"default": 16, "min": 2, "max": 150, "step": 2}),
                "paste_src": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute"
    CATEGORY = f"{cv_category_path}/Misc"

    def compute(self, src, mask, kernel_size, paste_src):
        from comfy.model_management import is_nvidia

        # setup input data
        kernel_size += 1
        image = tensor2opencv(src, 3)
        mask = tensor2opencv(mask, 1)

        # setup kernel ( maybe add optional input later for custom kernel? )
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel = np.where(kernel > 0, 1, 0)

        #  resize mask if it's size doesn't match the image's
        if image.shape[0:2] != mask.shape[0:2]:
            print("MaskedOuterBlur node will resize mask to fit the image.")
            mask = cv.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv.INTER_LINEAR)

        # extend image borders so that the algorithm doesn't have to take them into account
        border_size = kernel_size // 2
        image_extended = cv.copyMakeBorder(image, border_size, border_size, border_size, border_size,
                                           cv.BORDER_REPLICATE)
        mask_extended = cv.copyMakeBorder(mask, border_size, border_size, border_size, border_size, cv.BORDER_REPLICATE)

        # convert the image to float32
        image_float32 = image_extended.astype('float32')
        mask_float32 = mask_extended.astype('float32')

        if is_nvidia():  # is this check legit?
            import cupy as cp
            from numba.cuda import get_current_device
            from .utils.mask_outer_blur import blur_cuda

            # setup cupy arrays
            image_cupy = cp.asarray(image_float32)
            mask_cupy = cp.asarray(mask_float32)
            # note: don't pass extended size here; more data than needed to retrieve from gpu.
            #       instead, rawkernel outputs the final directly with the kernel size in mind
            #       and there is no need to crop after the computation
            out = cp.zeros((mask.shape[0], mask.shape[1], 3), dtype=cp.float32)
            kernel_gpu = cp.asarray(kernel)

            # setup grid/block sizes
            gpu = get_current_device()
            w, h = mask.shape[1], mask.shape[0]
            block_dim_x, block_dim_y = np.floor(
                np.array([w / h, h / w]) * gpu.MAX_THREADS_PER_BLOCK ** (1 / 2)).astype(np.int32)
            gridx, gridy = np.ceil(np.array([w / block_dim_x, h / block_dim_y])).astype(np.int32)

            # run on gpu, and then fetch result as numpy array
            blur_cuda((gridx, gridy), (block_dim_x, block_dim_y),
                      (image_cupy, mask_cupy, out, kernel_gpu, kernel_size, mask_extended.shape[1],
                       mask_extended.shape[0]))
            result_float32 = cp.asnumpy(out)

        else:  # run on cpu
            from .utils.mask_outer_blur import blur_cpu
            result_float32 = blur_cpu(image_float32, mask_float32, kernel, kernel_size, mask_extended.shape[1],
                                      mask_extended.shape[0])
            # remove added borders ( this is not required in gpu version;
            #                        only done here to avoid computing two sets of coordinates for every pixel )
            result_float32 = result_float32[border_size:-border_size, border_size:-border_size, :]

        if paste_src:  # paste src onto result using mask
            indices = mask > 0
            result_float32[indices, :] = image[indices, :]

        result = opencv2tensor(result_float32)
        return (result,)

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
    "Hue Mode (InRange hsv)": IN_RANGE_HUE_MODE,
    "InRange (hsv)": InRangeHSV,
    "DistanceTransform": DistanceTransform,
    # note: invert already exist: should be named ImageInvert, unless "overwritten" by some other custom node

    "MorphologicOperation": MorphologicOperation,
    "MorphologicSkeletoning": MorphologicSkeletoning,

    "ColorDictionary": ColorDefaultDictionary,
    "ColorDictionary (custom)": ColorCustomDictionary,
    "FindComplementaryColor": FindComplementaryColor,
    "KMeansColor": KMeansColor,
    "NaiveAutoKMeansColor": NaiveAutoKMeansColor,
    "RGB to HSV": ColorToHSVColor,
    "SampleColorHSV": SampleColorHSV,
    "BuildColorRangeHSV (hsv)": BuildColorRangeHSV,
    "BuildColorRangeAdvanced (hsv)": BuildColorRangeHSVAdvanced,

    "Remap": Remap,
    "RemapToInnerCylinder": InnerCylinderRemap,
    "RemapToOuterCylinder": OuterCylinderRemap,
    "RemapPinch": RemapPinch,
    "RemapStretch": RemapStretch,
    "RemapBarrelDistortion": RemapBarrelDistortion,
    "RemapReverseBarrelDistortion": RemapReverseBarrelDistortion,
    "RemapInsideParabolas": RemapInsideParabolas,
    "RemapInsideParabolasAdvanced": RemapInsideParabolasAdvanced,
    "RemapFromInsideParabolas": RemapFromInsideParabolas,
    "RemapToQuadrilateral": RemapQuadrilateral,
    "RemapFromQuadrilateral (homography)": RemapFromQuadrilateral,
    "RemapWarpPolar": RemapWarpPolar,

    "MaskOuterBlur": MaskOuterBlur,
}
