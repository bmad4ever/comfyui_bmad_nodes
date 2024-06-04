import numpy as np
import math
import nodes
import torch
from .utils.dry import (base_category_path, images_category_path, conditioning_category_path,
                        opencv2tensor, tensor2opencv, image_output_formats_options, image_output_formats_options_map,
                        grid_len_INPUT, maybe_convert_img, rect_modes, rect_modes_map,
                        prepare_text_for_eval, get_arg_name_from_multiple_inputs, print_yellow)
from .utils.color import ColorClip, color255_INPUT


lists_category_path = f"{base_category_path}/Lists"
latent_category_path = f"{base_category_path}/latent"


class StringNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"inStr": ("STRING", {"default": ""})}, }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "pass_it"
    CATEGORY = base_category_path

    def pass_it(self, inStr):
        return (inStr,)


class ColorClipSimple(ColorClip):
    @classmethod
    def INPUT_TYPES(cls):
        return super().get_types(advanced=False)

    def color_clip(self, image, color, target, complement):
        image = self.clip(image, color, target, complement)
        return (image,)


class ColorClipAdvanced(ColorClip):
    @classmethod
    def INPUT_TYPES(cls):
        return super().get_types(advanced=True)

    def color_clip(self, image, color, target, complement, color_a=None, color_b=None):
        image = self.clip(image, color, target, complement, color_a, color_b)
        return (image,)


class MonoMerge:
    target = ["white", "black"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "target": (cls.target, {"default": "white"}),
                "output_format": (image_output_formats_options, {
                    "default": image_output_formats_options[0]
                })
                ,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "monochromatic_merge"
    CATEGORY = images_category_path

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

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",),
                             "columns": grid_len_INPUT,
                             "rows": grid_len_INPUT,
                             }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "repeat_into_grid"
    CATEGORY = latent_category_path

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

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "columns": grid_len_INPUT,
                             "rows": grid_len_INPUT,
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat_into_grid"
    CATEGORY = images_category_path

    def repeat_into_grid(self, image, columns, rows):
        samples = image.movedim(-1, 1)
        samples = samples.repeat(1, 1, rows, columns)
        samples = samples.movedim(1, -1)
        return (samples,)


class UnGridImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "columns": grid_len_INPUT,
                             "rows": grid_len_INPUT,
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ungridify"
    CATEGORY = images_category_path
    OUTPUT_IS_LIST = (True,)

    def ungridify(self, image, columns, rows):
        tiles = []
        samples = image.movedim(-1, 1)
        _, _, height, width = samples.size()
        tile_height = height // rows
        tile_width = width // columns

        for y in range(0, rows * tile_height, tile_height):
            for x in range(0, columns * tile_width, tile_width):
                tile = samples[:, :, y:y + tile_height, x:x + tile_width]
                tile = tile.movedim(1, -1)
                tiles.append(tile)

        return (tiles,)


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

    @classmethod
    def INPUT_TYPES(cls):
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
    CATEGORY = conditioning_category_path

    def set_conditioning(self, base, columns, rows, width, height, strength, **kwargs):
        cond = base
        cond_set_area_node = nodes.ConditioningSetArea()
        cond_combine_node = nodes.ConditioningCombine()

        for r in range(rows):
            for c in range(columns):
                arg_name = f"r{r + 1}_c{c + 1}"
                new_cond = kwargs[arg_name]
                new_cond_area = cond_set_area_node.append(new_cond, width, height, c * width, r * height, strength)[0]
                new_cond = cond_combine_node.combine(new_cond_area, cond)[0]

                cond = new_cond
        return (cond,)


class ConditioningGridStr:
    """
    Node similar to ConditioningGridCond, but automates an additional step, using a ClipTextEncode per text input.
    Each conditioning obtained from the text inputs is then used as input for the Grid's AreaConditioners.
    """

    @classmethod
    def INPUT_TYPES(cls):
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
    CATEGORY = conditioning_category_path

    def set_conditioning(self, clip, base, columns, rows, width, height, strength, **kwargs):
        text_encode_node = nodes.CLIPTextEncode()
        cond_grid_node = ConditioningGridCond()

        encoded_base = text_encode_node.encode(clip, base)[0]
        encoded_grid = {}
        for r in range(rows):
            for c in range(columns):
                cell = f"r{r + 1}_c{c + 1}"
                encoded_grid[cell] = text_encode_node.encode(clip, kwargs[cell])[0]

        return cond_grid_node.set_conditioning(encoded_base, columns, rows, width, height, strength, **encoded_grid)


class CombineMultipleConditioning:
    """
    Node to save space and time combining multiple conditioning nodes.

    Set the number of cond inputs to combine in "combine" and then
    call "update inputs" menu option to set the given number of input sockets.
    """

    # TODO: consider implementing similar node for gligen

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "combine": ("INT", {"default": 3, "min": 2, "max": 50, "step": 1}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine_conds"
    CATEGORY = conditioning_category_path

    def combine_conds(self, combine, **kwargs):
        cond_combine_node = nodes.ConditioningCombine()

        cond = kwargs["c1"]
        for c in range(1, combine):
            new_cond = kwargs[f"c{c + 1}"]
            cond = cond_combine_node.combine(new_cond, cond)[0]

        return (cond,)


class CombineMultipleSelectiveConditioning:
    """
    Similar to CombineMultipleConditioning, but allows to specify the set of inputs to be combined.
    I.e. some inputs may be discarded and not contribute to the output.

    The "to_use" is a list of indexes of the inputs to use.
    """

    # TODO: consider implementing similar node for gligen

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "to_use": ("INT_ARRAY",),
            "combine": ("INT", {"default": 2, "min": 2, "max": 50, "step": 1}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine_conds"
    CATEGORY = conditioning_category_path

    def combine_conds(self, to_use, **kwargs):
        cond_combine_node = nodes.ConditioningCombine()

        to_use = to_use.copy()
        cond = kwargs[f"c{to_use.pop(0)}"]
        if len(to_use) == 0:
            return (cond,)

        for i in to_use:
            new_cond = kwargs[f"c{i}"]
            cond = cond_combine_node.combine(new_cond, cond)[0]

        return (cond,)


class AddString2Many:
    """
    Append or prepend a string to other, many, strings.
    """

    OPERATION = ["append", "prepend"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "to_add": ("STRING", {"default": '', "multiline": False}),
            "inputs_len": ("INT", {"default": 3, "min": 2, "max": 32, "step": 1}),
            "operation": (cls.OPERATION, {"default": 'append'}),
        }}

    RETURN_TYPES = tuple(["STRING" for x in range(32)])
    FUNCTION = "add_str"
    CATEGORY = conditioning_category_path

    def add_str(self, to_add, inputs_len, operation, **kwargs):
        new_strs = []
        for r in range(inputs_len):
            str_input_name = f"i{r + 1}"
            new_str = kwargs[str_input_name]
            if operation == "append":
                new_str = new_str + to_add
            else:
                new_str = to_add + new_str
            new_strs.append(new_str)

        return tuple(new_strs)


class AdjustRect:
    round_mode_map = {
        'Floor': math.floor,  # may be close to the image's edge, keep rect tight
        'Ceil': math.ceil,  # need the margin and image's edges aren't near
        'Round': round,  # whatever fits closest to the original rect
        'Exact': lambda v: 1  # force exact measurement
    }
    round_modes = list(round_mode_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "a": ("INT", {"min": 0, "max": np.iinfo(np.int32).max, "step": 1}),
            "b": ("INT", {"min": 0, "max": np.iinfo(np.int32).max, "step": 1}),
            "c": ("INT", {"min": 0, "max": np.iinfo(np.int32).max, "step": 1}),
            "d": ("INT", {"min": 0, "max": np.iinfo(np.int32).max, "step": 1}),
            "xm": ("INT", {"default": 64, "min": 2, "max": 1280, "step": 2}),
            "ym": ("INT", {"default": 64, "min": 2, "max": 1280, "step": 2}),
            "round_mode": (cls.round_modes, {"default": cls.round_modes[2]}),
            "input_format": (rect_modes, {"default": rect_modes[1]}),
            "output_format": (rect_modes, {"default": rect_modes[1]}),
        }}

    RETURN_TYPES = tuple(["INT" for x in range(4)])
    FUNCTION = "adjust"
    CATEGORY = base_category_path

    def adjust(self, a, b, c, d, xm, ym, round_mode, input_format, output_format):
        x1, y1, x2, y2 = rect_modes_map[input_format]["toBounds"](a, b, c, d)
        center_x = (x1 + x2) // 2 + 1
        center_y = (y1 + y2) // 2 + 1
        # reasoning:
        # 00 01 02 03 04 05
        # __ -- -- -- -- __ ( original len of 4 )
        # __ x1 __ cx __ x2 ( target len of 4   )
        # most crop implementations include x1 but exclude x2;
        # thus is closer to original input
        # note that xm and ym are always even

        half_new_len_x = self.round_mode_map[round_mode]((x2 - x1) / xm) * xm // 2
        half_new_len_y = self.round_mode_map[round_mode]((y2 - y1) / ym) * ym // 2

        # note: these points can fall outside the image space
        x2 = x1 = center_x
        x2 += half_new_len_x
        x1 -= half_new_len_x
        y2 = y1 = center_y
        y2 += half_new_len_y
        y1 -= half_new_len_y

        # convert to desired output format
        x1, y1, x2, y2 = rect_modes_map[output_format]["fromBounds"](x1, y1, x2, y2)

        return (x1, y1, x2, y2,)


class VAEEncodeBatch:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "inputs_len": ("INT", {"default": 3, "min": 2, "max": 32, "step": 1}),
            "vae": ("VAE",)
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = base_category_path

    def encode(self, inputs_len, vae, **kwargs):
        vae_encoder = nodes.VAEEncode()

        def get_latent(input_name):
            pixels = kwargs[input_name]
            pixels = vae_encoder.vae_encode_crop_pixels(pixels)
            return vae.encode(pixels[:, :, :, :3])

        latent = get_latent("image_1")
        for r in range(1, inputs_len):
            latent = torch.cat([latent, get_latent(f"image_{r + 1}")], dim=0)

        return ({"samples": latent},)


class AnyToAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "v": ("*",),
            "function": ("STRING", {"multiline": True, "default": ""}),
        }}

    FUNCTION = "eval_it"
    CATEGORY = f"{base_category_path}/⚠️⚠️⚠️"
    RETURN_TYPES = tuple(["*" for x in range(16)])

    def eval_it(self, v, function):
        function = prepare_text_for_eval(function)
        expression = eval(f"lambda v: {function}", {
            "__builtins__": {},
            "tuple": tuple, "list": list},
                          {})
        result = expression(v)
        return result


class MaskGridNKSamplersAdvanced(nodes.KSamplerAdvanced):
    fork_before_sampling = {
        "Sample then Fork": False,
        "Fork then Sample": True
    }
    fork_options = list(fork_before_sampling.keys())

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["mask"] = ("IMAGE",)
        types["required"]["rows"] = ("INT", {"default": 1, "min": 1, "max": 16})
        types["required"]["columns"] = ("INT", {"default": 3, "min": 1, "max": 16})
        types["required"]["mode"] = (cls.fork_options, {"default": cls.fork_options[0]})
        return types

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "gen_batch"
    CATEGORY = f"{base_category_path}/experimental"

    def gen_batch(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                  latent_image, start_at_step, end_at_step, return_with_leftover_noise,
                  mask, rows, columns, mode, denoise=1.0):

        # setup sizes
        _, _, latent_height_as_img, latent_width_as_img = latent_image['samples'].size()
        latent_width_as_img *= 8
        latent_height_as_img *= 8
        _, mask_height, mask_width, _ = mask.size()

        # existing nodes required for the operation
        set_mask_node = nodes.SetLatentNoiseMask()

        latents = []

        if not self.fork_before_sampling[mode]:
            # FORK AFTER SAMPLING

            # prepare mask
            mask = RepeatIntoGridImage().repeat_into_grid(mask, columns, rows)[0]
            new_mask = torch.zeros((latent_height_as_img, latent_width_as_img))
            new_mask[:, :] = mask[0, :, :, 0]

            # prepare latent w/ mask and send to ksampler
            sampled_latent = set_mask_node.set_mask(samples=latent_image, mask=new_mask)[0]
            sampled_latent = \
            super().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                           sampled_latent, start_at_step, end_at_step, return_with_leftover_noise, denoise)[0][
                'samples']

            # adjust mask sizes for latent space
            mask_height //= 8
            mask_width //= 8

            # fork and copy regions from original latent
            for r in range(rows):
                for c in range(columns):
                    x2 = x1 = mask_width * c
                    x2 += mask_width
                    y2 = y1 = mask_height * r
                    y2 += mask_height
                    new_latent = latent_image['samples'].clone()
                    new_latent[0, :, y1:y2, x1:x2] = sampled_latent[0, :, y1:y2, x1:x2]
                    latents.append(new_latent)  # add new latent
        else:
            # FORK BEFORE SAMPLING
            for r in range(rows):
                for c in range(columns):
                    # copy source mask to a new empty mask
                    new_mask = torch.zeros((latent_height_as_img, latent_width_as_img))
                    new_mask[mask_height * r:mask_height * (r + 1), mask_width * c:mask_width * (c + 1)] = mask[0, :, :,
                                                                                                           0]

                    # prepare latent w/ mask and send to ksampler
                    new_latent = set_mask_node.set_mask(samples=latent_image.copy(), mask=new_mask)[0]
                    new_latent = \
                    super().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive,
                                   negative,
                                   new_latent, start_at_step, end_at_step, return_with_leftover_noise, denoise)[0][
                        'samples']

                    latents.append(new_latent)  # add new latent

        return ({"samples": torch.cat([batch for batch in latents], dim=0)},)


class MergeLatentsBatchGridwise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "batch": ("LATENT",),
            "mask": ("IMAGE",),  # only to fetch the sizes, not really needed.
            "rows": ("INT", {"default": 1, "min": 1, "max": 16}),
            "columns": ("INT", {"default": 1, "min": 1, "max": 16})
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "merge"
    CATEGORY = latent_category_path

    def merge(self, batch, mask, rows, columns):
        _, mask_height, mask_width, _ = mask.size()
        mask_height //= 8
        mask_width //= 8
        _, cs, hs, ws = batch["samples"].size()
        print(f'{batch["samples"].size()}')
        merged = torch.empty(size=(1, cs, hs, ws), dtype=batch["samples"].dtype, device=batch["samples"].device)
        for r in range(rows):
            for c in range(columns):
                x2 = x1 = mask_width * c
                x2 += mask_width
                y2 = y1 = mask_height * r
                y2 += mask_height
                merged[0, :, y1:y2, x1:x2] = batch["samples"][c + r * columns, :, y1:y2, x1:x2]

        return ({"samples": merged},)


# ===================================================

# region cond lists

class CLIPEncodeMultiple(nodes.CLIPTextEncode):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "inputs_len": ("INT", {"default": 9, "min": 0, "max": 32}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "gen2"
    CATEGORY = conditioning_category_path
    OUTPUT_IS_LIST = (True,)

    def gen2(self, clip, inputs_len, **kwargs):
        conds = []
        for i in range(inputs_len):
            arg_name = get_arg_name_from_multiple_inputs("string", i)
            conds.append(super().encode(clip, kwargs[arg_name])[0])
        return (conds,)


class ControlNetHadamard(nodes.ControlNetApply):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conds": ("CONDITIONING",),
                             "control_net": ("CONTROL_NET",),
                             "image": ("IMAGE",),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = conditioning_category_path
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def apply(self, conds, control_net, images, strength):
        control_net = control_net[0]
        strength = strength[0]

        assert len(images) == len(conds), "lists sizes do not match"

        print(len(images))
        print(len(images[0]))
        print(len(conds))
        new_conds = []
        for i in range(len(images)):
            new_conds.append(super().apply_controlnet(conds[i], control_net, images[i], strength)[0])
        return (new_conds,)


class ControlNetHadamardManual(ControlNetHadamard):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conds": ("CONDITIONING",),
                             "control_net": ("CONTROL_NET",),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "inputs_len": ("INT", {"default": 9, "min": 0, "max": 32})
                             }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = conditioning_category_path
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def apply(self, conds, control_net, strength, inputs_len, **kwargs):
        inputs_len = inputs_len[0]
        images = []
        for i in range(inputs_len):
            arg_name = get_arg_name_from_multiple_inputs("image", i)
            images.append(kwargs[arg_name][0])
        return super().apply(conds, control_net, images, strength)


# endregion cond lists workflow


class FlatLatentsIntoSingleGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latents": ("LATENT",), }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "flat_into_grid"
    CATEGORY = latent_category_path

    def flat_into_grid(self, latents):
        n, lc, lh, lw = latents['samples'].size()
        length_in_tiles = math.ceil(math.sqrt(n))
        new_latent = torch.zeros((1, lc, lh * math.ceil(n / length_in_tiles), lw * length_in_tiles),
                                 dtype=latents["samples"].dtype, device=latents["samples"].device)
        r = c = 0  # row and column indexes
        for i in range(n):
            x1 = x2 = lw * c
            x2 += lw
            y1 = y2 = lh * r
            y2 += lh
            new_latent[0, :, y1:y2, x1:x2] = latents["samples"][i, :, :, :]
            c += 1
            if c >= length_in_tiles:
                c = 0
                r += 1

        return ({"samples": new_latent},)


class ColorRGB:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"r": color255_INPUT, "g": color255_INPUT, "b": color255_INPUT}}

    RETURN_TYPES = ("COLOR",)
    FUNCTION = "ret"
    CATEGORY = images_category_path

    def ret(self, r, g, b):
        return ([r, g, b],)


class ColorRGBFromHex:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"hex": ("STRING", {"default": "#000000"})}}

    RETURN_TYPES = ("COLOR",)
    FUNCTION = "ret"
    CATEGORY = images_category_path

    def ret(self, hex):
        import re
        hex_color_pattern = r'^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$'
        if re.match(hex_color_pattern, hex) is None:
            print_yellow(f"ColorRGBFromHex node > The following is not a valid hex code:{hex}")
        return (hex,)


class ImageBatchToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "to_list"
    CATEGORY = images_category_path
    OUTPUT_IS_LIST = (True,)

    def to_list(self, images):
        image_list = [images[i][None, ...] for i in range(images.shape[0])]
        return (image_list,)


# region  get items from list

class UnMakeListMeta(type):
    def __new__(cls, name, bases, attrs):
        if 'RETURN_TYPES' not in attrs:
            attrs['RETURN_TYPES'] = tuple([attrs["TYPE"].upper() for _ in range(32)])

        if 'CATEGORY' not in attrs:
            attrs['CATEGORY'] = f'{lists_category_path}/GetAll'

        attrs['FUNCTION'] = 'get_all'
        attrs['INPUT_IS_LIST'] = True

        def get_all(self, list):
            return tuple(list)

        def INPUT_TYPES(cls):
            return {
                "required": {
                    "list": (attrs["TYPE"].upper(), {"forceInput": True})
                }
            }

        attrs['get_all'] = get_all

        if 'INPUT_TYPES' not in attrs:
            attrs['INPUT_TYPES'] = classmethod(INPUT_TYPES)

        return super().__new__(cls, name, bases, attrs)


class GetSingleFromListMeta(type):
    """
    Allows random access too using primitive node!
    Can also use negative indexes to access in reverse.
    """

    def __new__(cls, name, bases, attrs):
        if 'RETURN_TYPES' not in attrs:
            attrs['RETURN_TYPES'] = (attrs["TYPE"].upper(),)

        if 'CATEGORY' not in attrs:
            attrs['CATEGORY'] = f'{lists_category_path}/Get1'

        attrs['FUNCTION'] = 'get_one'
        attrs['INPUT_IS_LIST'] = True

        def get_one(self, list, index):
            index = index[0]
            index = index % len(list)
            return (list[index],)

        def INPUT_TYPES(cls):
            return {
                "required": {
                    "list": (attrs["TYPE"].upper(), {"forceInput": True}),
                    "index": ("INT", {"default": 0, "min": -2147483648})
                }
            }

        attrs['get_one'] = get_one

        if 'INPUT_TYPES' not in attrs:
            attrs['INPUT_TYPES'] = classmethod(INPUT_TYPES)

        return super().__new__(cls, name, bases, attrs)


class FromListGetMasks(metaclass=UnMakeListMeta):  TYPE = "MASK"
class FromListGetImages(metaclass=UnMakeListMeta):  TYPE = "IMAGE"
class FromListGetLatents(metaclass=UnMakeListMeta):  TYPE = "LATENT"
class FromListGetConds(metaclass=UnMakeListMeta):  TYPE = "CONDITIONING"
class FromListGetModels(metaclass=UnMakeListMeta):  TYPE = "MODEL"
class FromListGetColors(metaclass=UnMakeListMeta):  TYPE = "COLOR"
class FromListGetStrings(metaclass=UnMakeListMeta): TYPE = "STRING"
class FromListGetInts(metaclass=UnMakeListMeta): TYPE = "INT"
class FromListGetFloats(metaclass=UnMakeListMeta): TYPE = "FLOAT"


class FromListGet1Mask(metaclass=GetSingleFromListMeta):  TYPE = "MASK"
class FromListGet1Image(metaclass=GetSingleFromListMeta):  TYPE = "IMAGE"
class FromListGet1Latent(metaclass=GetSingleFromListMeta):  TYPE = "LATENT"
class FromListGet1Cond(metaclass=GetSingleFromListMeta):  TYPE = "CONDITIONING"
class FromListGet1Model(metaclass=GetSingleFromListMeta):  TYPE = "MODEL"
class FromListGet1Color(metaclass=GetSingleFromListMeta):  TYPE = "COLOR"
class FromListGet1String(metaclass=GetSingleFromListMeta): TYPE = "STRING"
class FromListGet1Int(metaclass=GetSingleFromListMeta): TYPE = "INT"
class FromListGet1Float(metaclass=GetSingleFromListMeta): TYPE = "FLOAT"


# TODO could a IntBatch be of use? e.g. to fetch multiple ranges from a list


# endregion


# region create list from multiple single inputs

class MakeListMeta(type):
    def __new__(cls, name, bases, attrs):
        if 'RETURN_TYPES' not in attrs:
            attrs['RETURN_TYPES'] = (attrs["TYPE"].upper(),)

        if 'CATEGORY' not in attrs:
            attrs['CATEGORY'] = f'{lists_category_path}/Make or Intercalate'

        attrs['FUNCTION'] = 'to_list'
        attrs['OUTPUT_IS_LIST'] = (True,)

        def to_list(self, inputs_len, **kwargs):
            list = []
            for i in range(inputs_len):
                arg_name = get_arg_name_from_multiple_inputs(self.TYPE.lower(), i)
                list.append(kwargs[arg_name])
            return (list,)

        def INPUT_TYPES(cls):
            return {"required": {
                "inputs_len": ("INT", {"default": 2, "min": 0, "max": 32}),
            }}

        if 'to_list' not in attrs:
            attrs['to_list'] = to_list
        attrs['INPUT_TYPES'] = classmethod(INPUT_TYPES)

        return super().__new__(cls, name, bases, attrs)


class ExtendListMeta(MakeListMeta):
    def __new__(cls, name, bases, attrs):
        def to_list(self, inputs_len, **kwargs):
            list = []
            for i in range(inputs_len[0]):
                arg_name = get_arg_name_from_multiple_inputs(self.TYPE.lower(), i)
                list.extend(kwargs[arg_name])
            return (list,)

        attrs['INPUT_IS_LIST'] = True
        attrs['to_list'] = to_list
        attrs['CATEGORY'] = f'{lists_category_path}/Extend'

        return super().__new__(cls, name, bases, attrs)




class ToMaskList(metaclass=MakeListMeta): TYPE = "MASK"
class ToImageList(metaclass=MakeListMeta): TYPE = "IMAGE"
class ToLatentList(metaclass=MakeListMeta): TYPE = "LATENT"
class ToCondList(metaclass=MakeListMeta): TYPE = "CONDITIONING"
class ToModelList(metaclass=MakeListMeta): TYPE = "MODEL"
class ToColorList(metaclass=MakeListMeta): TYPE = "COLOR"
class ToStringList(metaclass=MakeListMeta): TYPE = "STRING"
class ToIntList(metaclass=MakeListMeta): TYPE = "INT"
class ToFloatList(metaclass=MakeListMeta): TYPE = "FLOAT"


class ExtendMaskList(metaclass=ExtendListMeta): TYPE = "MASK"
class ExtendImageList(metaclass=ExtendListMeta): TYPE = "IMAGE"
class ExtendLatentList(metaclass=ExtendListMeta): TYPE = "LATENT"
class ExtendCondList(metaclass=ExtendListMeta): TYPE = "CONDITIONING"
class ExtendModelList(metaclass=ExtendListMeta): TYPE = "MODEL"
class ExtendColorList(metaclass=ExtendListMeta): TYPE = "COLOR"
class ExtendStringList(metaclass=ExtendListMeta): TYPE = "STRING"
class ExtendIntList(metaclass=ExtendListMeta): TYPE = "INT"
class ExtendFloatList(metaclass=ExtendListMeta): TYPE = "FLOAT"

# endregion


NODE_CLASS_MAPPINGS = {
    "String": StringNode,
    "Add String To Many": AddString2Many,

    "Color (RGB)": ColorRGB,
    "Color (hexadecimal)": ColorRGBFromHex,
    "Color Clip": ColorClipSimple,
    "Color Clip (advanced)": ColorClipAdvanced,
    "MonoMerge": MonoMerge,

    "Repeat Into Grid (latent)": RepeatIntoGridLatent,
    "Repeat Into Grid (image)": RepeatIntoGridImage,
    "UnGridify (image)": UnGridImage,

    "Conditioning Grid (cond)": ConditioningGridCond,
    "Conditioning Grid (string)": ConditioningGridStr,
    # "Conditioning (combine multiple)": CombineMultipleConditioning, (missing javascript)
    # "Conditioning (combine selective)": CombineMultipleSelectiveConditioning (missing javascript),

    "AdjustRect": AdjustRect,

    "VAEEncodeBatch": VAEEncodeBatch,

    "AnyToAny": AnyToAny,

    "MaskGrid N KSamplers Advanced": MaskGridNKSamplersAdvanced,
    "Merge Latent Batch Gridwise": MergeLatentsBatchGridwise,

    "CLIPEncodeMultiple": CLIPEncodeMultiple,
    "ControlNetHadamard": ControlNetHadamard,
    "ControlNetHadamard (manual)": ControlNetHadamardManual,

    "FlatLatentsIntoSingleGrid": FlatLatentsIntoSingleGrid,

    "ImageBatchToList": ImageBatchToList,

    "FromListGetMasks": FromListGetMasks,
    "FromListGetImages": FromListGetImages,
    "FromListGetConds": FromListGetConds,
    "FromListGetLatents": FromListGetLatents,
    "FromListGetModels": FromListGetModels,
    "FromListGetColors": FromListGetColors,
    "FromListGetStrings": FromListGetStrings,
    "FromListGetInts": FromListGetInts,
    "FromListGetFloats": FromListGetFloats,
    "FromListGet1Mask": FromListGet1Mask,
    "FromListGet1Image": FromListGet1Image,
    "FromListGet1Latent": FromListGet1Latent,
    "FromListGet1Cond": FromListGet1Cond,
    "FromListGet1Model": FromListGet1Model,
    "FromListGet1Color": FromListGet1Color,
    "FromListGet1String": FromListGet1String,
    "FromListGet1Int": FromListGet1Int,
    "FromListGet1Float": FromListGet1Float,
    "ToMaskList": ToMaskList,
    "ToImageList": ToImageList,
    "ToLatentList": ToLatentList,
    "ToCondList": ToCondList,
    "ToModelList": ToModelList,
    "ToColorList": ToColorList,
    "ToStringList": ToStringList,
    "ToIntList": ToIntList,
    "ToFloatList": ToFloatList,
    "ExtendMaskList": ExtendMaskList,
    "ExtendImageList": ExtendImageList,
    "ExtendLatentList": ExtendLatentList,
    "ExtendCondList": ExtendCondList,
    "ExtendModelList": ExtendModelList,
    "ExtendColorList": ExtendColorList,
    "ExtendStringList": ExtendStringList,
    "ExtendIntList": ExtendIntList,
    "ExtendFloatList": ExtendFloatList,
}
