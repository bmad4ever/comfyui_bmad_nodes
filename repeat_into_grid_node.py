import torch
import numpy as np
from PIL import Image, ImageEnhance
from nodes import *


class RepeatIntoGrid:
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

    CATEGORY = "latent"

    def repeat_into_grid(self, samples, columns, rows):
        s = samples.copy()
        samples = samples['samples']
        tiled_samples = samples.repeat(1, 1, columns, rows)
        s['samples'] = tiled_samples
        return (s,)


NODE_CLASS_MAPPINGS = {
    "Repeat Into Grid": RepeatIntoGrid
}