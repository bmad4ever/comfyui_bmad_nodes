import torch
import numpy as np
import cv2
from PIL import Image, ImageOps


# TODO these nodes return the mask, not the image with the background removed!
#       this is somewhat misleading. Consider changing the methods names.
#       ( but to what? GrabCutMask? FramedMaskGrabCutMask? ...)


class FramedMaskGrabCut:
    frame_options = ['FULL_FRAME', 'IGNORE_BOTTOM', 'IGNORE_TOP', 'IGNORE_RIGHT', 'IGNORE_LEFT', 'IGNORE_HORIZONTAL',
                     'IGNORE_VERTICAL']
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
                    "default": 'FULL_FRAME'
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grab_cut"

    CATEGORY = "Bmad/CV"

    def grab_cut(self, image, thresh, iterations, margin, frame_option):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = np.array(image)

        thresh = 255. * thresh[0].cpu().numpy()
        thresh = Image.fromarray(np.clip(thresh, 0, 255).astype(np.uint8))
        thresh = ImageOps.grayscale(thresh)
        thresh = np.array(thresh)

        fg_model = np.zeros((1, 65), dtype="float")
        bg_model = np.zeros((1, 65), dtype="float")
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[thresh == 255] = cv2.GC_PR_FGD  # probable foreground

        # check what borders should be painted
        frame_option = self.frame_options_values[frame_option]
        include_bottom = not (frame_option & self.frame_options_values['IGNORE_BOTTOM'])
        include_top = not (frame_option & self.frame_options_values['IGNORE_TOP'])
        include_right = not (frame_option & self.frame_options_values['IGNORE_RIGHT'])
        include_left = not (frame_option & self.frame_options_values['IGNORE_LEFT'])

        # paint the borders as being background
        if include_bottom:
            mask[-margin:, :] = cv2.GC_BGD
        if include_top:
            mask[0:margin, :] = cv2.GC_BGD
        if include_right:
            mask[:, -margin:] = cv2.GC_BGD
        if include_left:
            mask[:, 0:margin] = cv2.GC_BGD

        mask, bg_model, fg_model = cv2.grabCut(image, mask, None, bg_model, fg_model, iterCount=iterations,
                                             mode=cv2.GC_INIT_WITH_MASK)

        # generate mask with "pixels" classified as background/foreground
        output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
                              0, 1)

        output_mask = (output_mask * 255).astype("uint8")
        image = Image.fromarray(output_mask.astype(np.uint8)).convert("RGB")
        # to truly grabcut instead of returning the output_mask as an image
        # comment the lines above this comment, and uncomment the lines below
        # image = cv2.bitwise_and(image, image, mask=outputMask)
        # image = Image.fromarray(image.astype(np.uint8)).convert("RGB")

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)


# ====================================================================================

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
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grab_cut"

    CATEGORY = "Bmad/CV"

    def grab_cut(self, image, iterations, x1, y1, x2, y2):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = np.array(image)

        fg_model = np.zeros((1, 65), dtype="float")
        bg_model = np.zeros((1, 65), dtype="float")
        mask = np.zeros(image.shape[:2], dtype="uint8")
        rect = (x1, y1, x2, y2)

        mask, bg_model, fg_model = cv2.grabCut(image, mask, rect, bg_model,
                                             fg_model, iterCount=iterations, mode=cv2.GC_INIT_WITH_RECT)

        # generate mask with "pixels" classified as background/foreground
        output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
                              0, 1)
        output_mask = (output_mask * 255).astype("uint8")

        # get an image from the output mask
        image = Image.fromarray(output_mask.astype(np.uint8)).convert("RGB")
        # image = image[y1:y2, x1:x2] #add option whether to crop or not
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)


# ====================================================================================


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
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grab_cut"

    CATEGORY = "Bmad/CV"

    def grab_cut(self, image, thresh_maybe, thresh_sure, iterations, margin, frame_option):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = np.array(image)

        def mkt(thresh):
            thresh = 255. * thresh[0].cpu().numpy()
            thresh = Image.fromarray(np.clip(thresh, 0, 255).astype(np.uint8))
            thresh = ImageOps.grayscale(thresh)
            return np.array(thresh)

        thresh_maybe = mkt(thresh_maybe)
        thresh_sure = mkt(thresh_sure)

        fg_model = np.zeros((1, 65), dtype="float")
        bg_model = np.zeros((1, 65), dtype="float")
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[thresh_sure == 0] = cv2.GC_PR_BGD  # probable background
        mask[thresh_maybe == 255] = cv2.GC_PR_FGD  # probable foreground
        mask[thresh_sure == 255] = cv2.GC_FGD  # foreground

        frame_option = self.frame_options_values[frame_option]
        include_bottom = not (frame_option & self.frame_options_values['IGNORE_BOTTOM'])
        include_top = not (frame_option & self.frame_options_values['IGNORE_TOP'])
        include_right = not (frame_option & self.frame_options_values['IGNORE_RIGHT'])
        include_left = not (frame_option & self.frame_options_values['IGNORE_LEFT'])

        if include_bottom:
            mask[-margin:, :] = cv2.GC_BGD 
        if include_top:
            mask[0:margin, :] = cv2.GC_BGD 
        if include_right:
            mask[:, -margin:] = cv2.GC_BGD  
        if include_left:
            mask[:, 0:margin] = cv2.GC_BGD  

        mask, bg_model, fg_model = cv2.grabCut(image, mask, None, bg_model, fg_model, iterCount=iterations,
                                               mode=cv2.GC_INIT_WITH_MASK)

        # generate mask with "pixels" classified as background/foreground
        output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
                               0, 1)
        output_mask = (output_mask * 255).astype("uint8")

        image = Image.fromarray(output_mask.astype(np.uint8)).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)


NODE_CLASS_MAPPINGS = {
    "Framed Mask Grab Cut": FramedMaskGrabCut,
    "Framed Mask Grab Cut 2": FramedMaskGrabCut2,
    "Rect Grab Cut": RectGrabCut
}
