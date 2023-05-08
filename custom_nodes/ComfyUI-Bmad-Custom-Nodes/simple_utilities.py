import torch
import numpy as np
from PIL import Image, ImageOps
from nodes import *
from .dry import ColorClip
from .dry import color255Input


class ColorClipRGB(ColorClip):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red": color255Input,
                "green": color255Input,
                "blue": color255Input,
                "target": (s.OPERATION, {"default": 'TO_WHITE'}),
                "complement": (s.OPERATION, {"default": 'TO_BLACK'})
            },
        }

    def color_clip(self, image, red, green, blue, target, complement):
        clip_color = (red, green, blue)
        image = self.clip(image, clip_color, target, complement)
        return (image,)


class MonoMerge:
    target = ["white", "black"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "target": (s.target, {"default": "white"})
                ,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "monochromatic_merge"

    CATEGORY = "Bmad/image"

    def monochromatic_merge(self, image1, image2, target):
        image1 = 255. * image1[0].cpu().numpy()
        image2 = 255. * image2[0].cpu().numpy()

        # Check if images have the same dimensions
        assert image1.shape == image2.shape, "Images must have the same dimensions"

        # Select the lesser L component at each pixel
        if target == "white":
            image = np.maximum(image1, image2)
        else:
            image = np.minimum(image1, image2)

        # Convert to PIL image
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image,)


class RepeatIntoGridLatent:
    """
    Tiles the input samples into a grid of configurable dimensions.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",),
                             "columns": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                             "rows": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                             }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "repeat_into_grid"

    CATEGORY = "Bmad/latent"

    def repeat_into_grid(self, samples, columns, rows):
        s = samples.copy()
        samples = samples['samples']
        tiled_samples = samples.repeat(1, 1, rows, columns)
        s['samples'] = tiled_samples
        return (s,)


class RepeatIntoGridImage:
    """
    Tiles the input samples into a grid of configurable dimensions.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "columns": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                             "rows": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat_into_grid"

    CATEGORY = "Bmad/image"

    def repeat_into_grid(self, image, columns, rows):
        samples = image.movedim(-1, 1)
        samples = samples.repeat(1, 1, rows, columns)
        samples = samples.movedim(1, -1)
        return (samples,)




NODE_CLASS_MAPPINGS = {
    "Color Clip RGB": ColorClipRGB,
    "MonoMerge": MonoMerge,
    "Repeat Into Grid (latent)": RepeatIntoGridLatent,
    "Repeat Into Grid (image)": RepeatIntoGridImage
}
