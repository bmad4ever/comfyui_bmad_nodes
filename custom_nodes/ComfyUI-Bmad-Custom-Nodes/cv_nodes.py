import cv2 as cv
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
        output_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),  0, 1)

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
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grab_cut"

    CATEGORY = "Bmad/CV/GrabCut"

    def grab_cut(self, image, thresh_maybe, thresh_sure, iterations,
                 margin, frame_option, binary_threshold, output_format):
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

        mask, bg_model, fg_model = cv.grabCut(image, mask, None, bg_model, fg_model, iterCount=iterations,
                                              mode=cv.GC_INIT_WITH_MASK)

        # generate mask with "pixels" classified as background/foreground
        output_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),  0, 1)
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

    RETURN_TYPES = ("CV_CONTOURS", "CV_CONTOURS_HIERARCHY" )
    FUNCTION = "find_contours"

    CATEGORY = "Bmad/CV/Contour"

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
                "contours": ("CV_CONTOURS",),
                "index_to_draw": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 1000,
                    "step": 1
                }),
                "color": ("COLOR", ),
                "thickness": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 32,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "draw"

    CATEGORY = "Bmad/CV/Contour"

    def draw(self, image, contours, index_to_draw, color, thickness):
        background = tensor2opencv(image)

        um_image = cv.UMat(background)
        cv.drawContours(um_image, contours, index_to_draw, ImageColor.getcolor(color, "RGB"), thickness)
        contour_image = um_image.get()

        image = opencv2tensor(contour_image)

        return (image, )


class GetContourFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "contours": ("CV_CONTOURS",),
                "index": ("INT", {"default": 0, "min": 0, "step":1})
            }
        }

    RETURN_TYPES = ("CV_CONTOUR", )
    FUNCTION = "get_contour"

    CATEGORY = "Bmad/CV/Contour"

    def get_contour(self, contours, index):
        if index >= len(contours):
            return (None, )
        return (contours[index],)


class ContourGetBoundingRect:

    def vanilla(self,x,y,w,h):
        return (x,y,w,h,)

    def corners(self,x,y,w,h):
        return (x, y, x + w, y + h,)

    def center_and_size(self, x, y, w, h):
        return (x + w // 2, y + h // 2, w, h,)

    def center_and_half_size(self, x, y, w, h):
        return (x + w // 2, y + h // 2, w // 2, h // 2,)

    rect_modes_map = {
        'top-left XY + WH': vanilla,
        'top-left XY + bottom-right XY': corners,
        'center XY (floored) + WH': center_and_size,
        'center XY + half WH (all floored)': center_and_half_size,
    }
    rect_modes = list(rect_modes_map.keys())

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "contour": ("CV_CONTOUR",),
                "return_mode": (s.rect_modes, {"default": s.rect_modes[1]})
            },
        }

    RETURN_TYPES = tuple(["INT" for x in range(4)])
    FUNCTION = "compute"

    CATEGORY = "Bmad/CV/Contour"

    def compute(self, contour, return_mode):
        if contour is None:
            print("Contour = None !")
            return (0,0,0,0, )

        return self.rect_modes_map[return_mode](self, *cv.boundingRect(contour))


# endregion contour nodes


NODE_CLASS_MAPPINGS = {
    "Framed Mask Grab Cut": FramedMaskGrabCut,
    "Framed Mask Grab Cut 2": FramedMaskGrabCut2,
    "Rect Grab Cut": RectGrabCut,

    "Contours": Contours,
    "Draw Contour(s)": DrawContours,
    "Get Contour from list": GetContourFromList,
    "BoundingRect (contours)": ContourGetBoundingRect,
}
