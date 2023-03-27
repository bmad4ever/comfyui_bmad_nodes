import torch
import numpy as np
import cv2
from PIL import Image, ImageOps

class OtsuThreshold:
    thresh_modes = ['BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    border_types = [
    #"BORDER_CONSTANT"	,
    "BORDER_REPLICATE"	,
    "BORDER_REFLECT"	,
    "BORDER_WRAP"		,
    "BORDER_REFLECT_101",
    "BORDER_TRANSPARENT",
    "BORDER_ISOLATED"	]
    
    modes = {
                'BINARY':		 cv2.THRESH_BINARY,
                'BINARY_INV':	 cv2.THRESH_BINARY_INV,
                'TRUNC':		 cv2.THRESH_TRUNC,
                'TOZERO':		 cv2.THRESH_TOZERO,
                'TOZERO_INV':    cv2.THRESH_TOZERO_INV,        
    }
    gaussian_border_types = {
                #"BORDER_CONSTANT": cv2.BORDER_CONSTANT,
                "BORDER_REPLICATE": cv2.BORDER_REPLICATE,
                "BORDER_REFLECT": cv2.BORDER_REFLECT,
                "BORDER_WRAP": cv2.BORDER_WRAP,
                "BORDER_REFLECT_101": cv2.BORDER_REFLECT_101,
                "BORDER_TRANSPARENT": cv2.BORDER_TRANSPARENT,
                "BORDER_ISOLATED": cv2.BORDER_ISOLATED
    }
    #channels = ["red", "green", "blue", "greyscale"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                #"channel": (s.channels, {"default": "greyscale"}),
                "thresh_mode": (s.thresh_modes, {"default": "BINARY"}),
                "gaussian_blur_x": ("INT", {
                    "default": 5, 
                    "min": 0,
                    "max": 200,
                    "step": 1
                }),
                    "gaussian_blur_y": ("INT", {
                    "default": 5, 
                    "min": 0,
                    "max": 200,
                    "step": 1
                }),
                "gaussian_border_type": (s.border_types, {"default": "BORDER_REPLICATE"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "otsu_thresthold"

    CATEGORY = "CV"
    

    def otsu_thresthold(self, image, thresh_mode, gaussian_blur_x, gaussian_blur_y, gaussian_border_type):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = ImageOps.grayscale(image)
        image = np.array(image)
        
        image = cv2.GaussianBlur(image,(gaussian_blur_x,gaussian_blur_y),self.gaussian_border_types[gaussian_border_type])
        _, image = cv2.threshold(image, 0, 255, self.modes[thresh_mode] + cv2.THRESH_OTSU)
        
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        return (image,)
    

NODE_CLASS_MAPPINGS = {
    "Otsu": OtsuThreshold
}