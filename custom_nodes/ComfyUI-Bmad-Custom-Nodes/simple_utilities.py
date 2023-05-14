import torch
import numpy as np
from PIL import Image, ImageOps

import nodes
from nodes import *
from .dry import *


class ColorClipRGB(ColorClip):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red": color255_INPUT,
                "green": color255_INPUT,
                "blue": color255_INPUT,
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
                             "columns": grid_len_INPUT,
                             "rows": grid_len_INPUT,
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
                             "columns": grid_len_INPUT,
                             "rows": grid_len_INPUT,
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat_into_grid"

    CATEGORY = "Bmad/image"

    def repeat_into_grid(self, image, columns, rows):
        samples = image.movedim(-1, 1)
        samples = samples.repeat(1, 1, rows, columns)
        samples = samples.movedim(1, -1)
        return (samples,)


class ConditioningGridCond:
    """
    Does the job of multiple area conditions of the same size adjacent to each other.
    Saves space, and is easier and quicker to set up and modify.


    Inputs related notes
    ----------
    base : conditioning
        for most cases, you can set the base from a ClipTextEncode with an empty string.
        If you wish to have something between the cells as common ground, lower the strength and set
        the base with the shared elements.
    columns and rows : integer
        In most cases should not be converted to input.

        dev note: I've considered disabling columns and rows options to convert to input
        on the javascript side, which (that I am aware) could be done with a modification
        to the core/WidgetInputs.js -> isConvertableWidget(...).
        However, upon reflection, I think there may be use cases in which the inputs are set for the
        maximum size but only a selected number of columns or rows given via input are used.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                             "columns": grid_len_INPUT,
                             "rows": grid_len_INPUT,
                             "width": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
                             "height": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
                             "strength": ("FLOAT", {"default": 3, }),
                             "base": ("CONDITIONING",)
                             }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_conditioning"

    CATEGORY = "Bmad/conditioning"

    def set_conditioning(self, base, columns, rows, width, height, strength, **kwargs):
        cond = base
        cond_set_area_node = nodes.ConditioningSetArea()
        cond_combine_node = nodes.ConditioningCombine()
        
        for r in range(rows):
            for c in range(columns):
                arg_name = f"r{r+1}_c{c+1}"
                new_cond = kwargs[arg_name]
                new_cond_area = cond_set_area_node.append(new_cond, width, height, c*width, r*height, strength)[0]
                new_cond = cond_combine_node.combine(new_cond_area, cond)[0]

                cond = new_cond
        return (cond,)


class ConditioningGridStr:
    """
    Node similar to ConditioningGridCond, but automates an additional step, using a ClipTextEncode per text input.
    Each conditioning obtained from the text inputs is then used as input for the Grid's AreaConditioners.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP",),
            "base": ("STRING", {"default": '', "multiline": False}),
            "columns": grid_len_INPUT,
            "rows": grid_len_INPUT,
            "width": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
            "height": ("INT", {"default": 256, "min": 16, "max": 2048, "step": 1}),
            "strength": ("FLOAT", {"default": 3, }),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_conditioning"

    CATEGORY = "Bmad/conditioning"

    def set_conditioning(self, clip, base, columns, rows, width, height, strength, **kwargs):
        text_encode_node = nodes.CLIPTextEncode()
        cond_grid_node = ConditioningGridCond()

        encoded_base = text_encode_node.encode(clip, base)[0]
        encoded_grid = {}
        for r in range(rows):
            for c in range(columns):
                cell = f"r{r+1}_c{c+1}"
                encoded_grid[cell] = text_encode_node.encode(clip, kwargs[cell])[0]

        return cond_grid_node.set_conditioning(encoded_base, columns, rows, width, height, strength, **encoded_grid)


NODE_CLASS_MAPPINGS = {
    "Color Clip RGB": ColorClipRGB,
    "MonoMerge": MonoMerge,
    "Repeat Into Grid (latent)": RepeatIntoGridLatent,
    "Repeat Into Grid (image)": RepeatIntoGridImage,
    "Conditioning Grid (cond)": ConditioningGridCond,
    "Conditioning Grid (string)": ConditioningGridStr,
}
