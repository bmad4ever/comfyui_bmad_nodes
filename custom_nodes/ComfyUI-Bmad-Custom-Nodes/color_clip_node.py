import torch
import numpy as np
from PIL import Image, ImageOps

import os
import sys

# TODO consider computing color distance and use a threshold
#  setting thresh to 1 should return the same as the current implementation

# TODO check if it is viable to have a color picker tool or
#  other alternative ways to input the color
#  Another useful alternative: select the segmentation item you want to filter
#  then it is mapped to the corresponding color


class ColorClip:


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "green": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1
                }),
                "blue": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
                    "step": 1
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_clip"

    CATEGORY = "Bmad/image"

    def color_clip(self, image, red, green, blue):
        target_rgb = (red, green, blue)

        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = np.array(image)

        # set all pixels with target color to zero (black)
        image[(image == target_rgb).all(axis=2)] = [0, 0, 0]

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)


NODE_CLASS_MAPPINGS = {
    "Color Clip": ColorClip
}
