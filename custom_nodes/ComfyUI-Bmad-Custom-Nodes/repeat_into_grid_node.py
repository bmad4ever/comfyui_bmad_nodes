import torch
import numpy as np
from PIL import Image, ImageEnhance
from nodes import *

import os
import sys

class RepeatIntoGridLatent:
    """
    Tiles the input samples into a grid of configurable dimensions.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
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
        return {"required": { "image": ("IMAGE",),
                              "columns": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                              "rows": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
                              }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat_into_grid"

    CATEGORY = "Bmad/image"

    def repeat_into_grid(self, image, columns, rows):
        samples = image.movedim(-1,1)
        samples = samples.repeat(1, 1, rows, columns)
        samples = samples.movedim(1,-1)
        return (samples,)


NODE_CLASS_MAPPINGS = {
    "Repeat Into Grid (latent)": RepeatIntoGridLatent,
    "Repeat Into Grid (image)": RepeatIntoGridImage
}