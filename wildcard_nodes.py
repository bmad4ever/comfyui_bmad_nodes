from .utils.dry import base_category_path
from .utils.quasi_safu import safe_eval

# there is no bypass for "*" here on purpose
# you either modify comfy to accept "*", or add the bypass here yourself

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
        return safe_eval(
            function,
            extra_names={'v': v},
            include_math=True,  # Gives access to 'm'
            use_compound_types=True,  # Allows list/tuple operations
            force_threading=True,
            timeout=120.0
        )


NODE_CLASS_MAPPINGS = {
    "AnyToAny": AnyToAny,
}
