import cv2 as cv
import numpy as np
from PIL import ImageColor
from .dry import *


# TODO these nodes return the mask, not the image with the background removed!
#       this is somewhat misleading. Consider changing the methods names.
#       ( but to what? GrabCutMask? FramedMaskGrabCutMask? ...)


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
        import sys
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
        image=image[0]
        index_to_draw=index_to_draw[0]
        color=color[0]
        thickness=thickness[0]

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

        fitness=fitness[0]
        select=select[0]

        # region prepare inputs
        if image is not None:
            image=image[0]
            image = tensor2opencv(image)
        if aux_contour is not None:
            aux_contour=aux_contour[0]

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
            intersection = cv2.bitwise_and(gray, cv2.drawContours(np.zeros_like(gray), [cnt], 0, 255, thickness=cv2.FILLED))
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

thresh_types_map = {
    'BINARY': cv.THRESH_BINARY,
    'BINARY_INV': cv.THRESH_BINARY_INV,
    'TRUNC': cv.THRESH_TRUNC,
    'TOZERO': cv.THRESH_TOZERO,
    'TOZERO_INV': cv.THRESH_TOZERO_INV,
}
thresh_types = list(thresh_types_map.keys())

border_types_map = {
    "BORDER_CONSTANT": cv.BORDER_CONSTANT,
    "BORDER_REPLICATE": cv.BORDER_REPLICATE,
    "BORDER_REFLECT": cv.BORDER_REFLECT,
    "BORDER_WRAP": cv.BORDER_WRAP,
    "BORDER_REFLECT_101": cv.BORDER_REFLECT_101,
    "BORDER_TRANSPARENT": cv.BORDER_TRANSPARENT,
    "BORDER_ISOLATED": cv.BORDER_ISOLATED
}
border_types = list(border_types_map.keys())


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
                "gaussian_border_type": (border_types, {"default": "BORDER_REPLICATE"}),
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
        } #TODO optional inputs


    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "search"
    CATEGORY = "Bmad/CV/Thresholding"

    def search(self, src, start_at, end_at, thresh_type ,downscale_factor, condition):
        import cv2
        import numpy
        import math

        o_img = tensor2opencv(src, 1)
        height, width = tuple(o_img.shape)
        img = cv.resize(o_img, ( height//downscale_factor, width//downscale_factor), interpolation=cv.INTER_AREA)

        max_v = max(start_at, end_at)
        min_v = min(start_at, end_at)
        range_to_check = range(min_v, max_v+1)
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
        return (img, )


#TODO maybe add InRange and GainDivision

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
        kernel = cv.getStructuringElement(self.kernel_types_map[kernel_type], (kernel_size_x+1, kernel_size_y+1))
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
        return (img, )

# endregion


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


NODE_CLASS_MAPPINGS = {
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
    # note: invert already exist: should be named ImageInvert, unless "overwritten" by some other custom node

    "ConvertImg": ConvertImg,

    "MorphologicOperation": MorphologicOperation,
    "MorphologicSkeletoning": MorphologicSkeletoning,
}
