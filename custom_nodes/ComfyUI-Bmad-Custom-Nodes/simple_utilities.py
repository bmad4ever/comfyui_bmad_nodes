from PIL import ImageColor

import nodes
from .dry import *


class StringNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"inStr": ("STRING", {"default": ""})}, }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "pass_it"
    CATEGORY = "Bmad"

    def pass_it(self, inStr):
        return (inStr,)


class ColorClip(ColorClip):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "color": ("COLOR",),
                "target": (s.OPERATION, {"default": 'TO_WHITE'}),
                "complement": (s.OPERATION, {"default": 'TO_BLACK'})
            },
        }

    def color_clip(self, image, color, target, complement):
        image = self.clip(image, ImageColor.getcolor(color, "RGB"), target, complement)
        return (image,)


class MonoMerge:
    target = ["white", "black"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "target": (s.target, {"default": "white"}),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
                ,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "monochromatic_merge"
    CATEGORY = "Bmad/image"

    def monochromatic_merge(self, image1, image2, target, output_format):
        image1 = tensor2opencv(image1, 1)
        image2 = tensor2opencv(image2, 1)

        # Select the lesser L component at each pixel
        if target == "white":
            image = np.maximum(image1, image2)
        else:
            image = np.minimum(image1, image2)

        image = maybe_convert_img(image, 1, image_output_formats_options_map[output_format])
        image = opencv2tensor(image)

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
        after setting the desired grid size, call the menu option "update inputs" to update
        the node's conditioning input sockets.

        In most cases, columns and rows, should not be converted to input.

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


class CombineMultipleConditioning:
    """
    Node to save space and time combining multiple conditioning nodes.

    Set the number of cond inputs to combine in "combine" and then
    call "update inputs" menu option to set the given number of input sockets.
    """
    # TODO: consider implementing similar node for gligen

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "combine": ("INT", {"default": 3, "min": 2, "max": 50, "step": 1}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine_conds"
    CATEGORY = "Bmad/conditioning"

    def combine_conds(self, combine, **kwargs):
        cond_combine_node = nodes.ConditioningCombine()

        cond = kwargs["c1"]
        for c in range(1, combine):
            new_cond = kwargs[f"c{c+1}"]
            cond = cond_combine_node.combine(new_cond, cond)[0]

        return (cond,)


class CombineMultipleSelectiveConditioning:
    """
    Similar to CombineMultipleConditioning, but allows to specify the set of inputs to be combined.
    I.e. some inputs may be discarded and not contribute to the output.

    The "to_use" is a list of indexes of the inputs to use.
    """

    # TODO: consider implementing similar node for gligen

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "to_use": ("INT_ARRAY", ),
            "combine": ("INT", {"default": 2, "min": 2, "max": 50, "step": 1}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine_conds"
    CATEGORY = "Bmad/conditioning"

    def combine_conds(self, to_use, **kwargs):
        cond_combine_node = nodes.ConditioningCombine()

        to_use = to_use.copy()
        cond = kwargs[f"c{to_use.pop(0)}"]
        if len(to_use) == 0:
            return (cond, )

        for i in to_use:
            new_cond = kwargs[f"c{i}"]
            cond = cond_combine_node.combine(new_cond, cond)[0]

        return (cond,)


class AddString2Many:
    """
    Append or prepend a string to other, many, strings.
    """

    OPERATION = ["append", "prepend"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "to_add": ("STRING", {"default": '', "multiline": False}),
            "inputs_len": ("INT", {"default": 3, "min": 2, "max": 32, "step": 1}),
            "operation": (s.OPERATION, {"default": 'append'}),
        }}

    RETURN_TYPES = tuple(["STRING" for x in range(32)])
    FUNCTION = "add_str"
    CATEGORY = "Bmad/conditioning"

    def add_str(self, to_add, inputs_len, operation, **kwargs):
        new_strs = []
        for r in range(inputs_len):
            str_input_name = f"i{r+1}"
            new_str = kwargs[str_input_name]
            if operation == "append":
                new_str = new_str + to_add
            else:
                new_str = to_add + new_str
            new_strs.append(new_str)

        return tuple(new_strs)


class AdjustRect:
    #TODO to be implemented
    round_mode_map = {
        'Floor': 1,  # may be close to the image's edge, keep rect tight
        'Ceil': 2,  # need the margin and image's edges aren't near
        'Round': 3,  # whatever fits closest to the original rect
    }
    round_modes = list(round_mode_map.keys())





NODE_CLASS_MAPPINGS = {
    "String": StringNode,

    "Color Clip": ColorClip,
    "MonoMerge": MonoMerge,

    "Repeat Into Grid (latent)": RepeatIntoGridLatent,
    "Repeat Into Grid (image)": RepeatIntoGridImage,

    "Conditioning Grid (cond)": ConditioningGridCond,
    "Conditioning Grid (string)": ConditioningGridStr,
    #"Conditioning (combine multiple)": CombineMultipleConditioning, (missing javascript)
    #"Conditioning (combine selective)": CombineMultipleSelectiveConditioning (missing javascript),

    "Add String To Many": AddString2Many
}
