from .utils.dry import *
from .simple_utilities import ConditioningGridCond, conditioning_category_path
from custom_nodes.ComfyUI_ADV_CLIP_emb.nodes import AdvancedCLIPTextEncode


class ConditioningGridStr_ADVEncode:
    """
    Node similar to ConditioningGridCond, but automates an additional step, using a ClipTextEncode per text input.
    Each conditioning obtained from the text inputs is then used as input for the Grid's AreaConditioners.
    """

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

            "token_normalization": (["none", "mean", "length", "length+mean"],),
            "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],)
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "set_conditioning"
    CATEGORY = conditioning_category_path

    #def encode(self, clip: comfy.sd.CLIP, text: str, parser: str, mean_normalization: bool, multi_conditioning: bool, use_old_emphasis_implementation: bool, use_CFGDenoiser:bool,with_SDXL=False,text_g="",text_l=""):
    def set_conditioning(self, clip, base, columns, rows, width, height, strength,
                         token_normalization, weight_interpretation,
                         **kwargs):
        text_encode_node = AdvancedCLIPTextEncode()
        cond_grid_node = ConditioningGridCond()

        encoded_base = text_encode_node.encode(clip, base, token_normalization, weight_interpretation, 'disable')[0]
        encoded_grid = {}
        for r in range(rows):
            for c in range(columns):
                cell = f"r{r + 1}_c{c + 1}"
                encoded_grid[cell] = \
                text_encode_node.encode(clip, kwargs[cell], token_normalization, weight_interpretation, 'disable')[0]

        return cond_grid_node.set_conditioning(encoded_base, columns, rows, width, height, strength, **encoded_grid)


class CLIPEncodeMultipleAdvanced(AdvancedCLIPTextEncode):
    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()  # TODO should refactor Grid class above to this too, so if original is changed, all the new options are added there too
        types["required"].pop("text")
        types["required"]["inputs_len"] = ("INT", {"default": 9, "min": 0, "max": 32})
        return types

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "gen2"
    CATEGORY = conditioning_category_path
    OUTPUT_IS_LIST = (True,)

    def gen2(self, clip, token_normalization, weight_interpretation, inputs_len, **kwargs):
        conds = []
        for i in range(inputs_len):
            arg_name = get_arg_name_from_multiple_inputs("string", i)
            conds.append(
                super().encode(clip, kwargs[arg_name], token_normalization, weight_interpretation, 'disable')[0])
        return (conds,)


NODE_CLASS_MAPPINGS = {
    "Conditioning Grid (string) Advanced": ConditioningGridStr_ADVEncode,
    "CLIPEncodeMultipleAdvanced": CLIPEncodeMultipleAdvanced
}
