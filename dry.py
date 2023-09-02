# D.R.Y ( Don't Repeat Yourself: cross file utilities )

from abc import abstractmethod
import torch
import numpy as np
from PIL import Image

color255_INPUT = ("INT", {
    "default": 0,
    "min": 0,
    "max": 255,
    "step": 1
})

grid_len_INPUT = ("INT", {
    "default": 3,
    "min": 1,
    "max": 8,
    "step": 1
})


class ColorClip:
    OPERATION = [
        "TO_BLACK",
        "TO_WHITE",
        "NOTHING"
    ]

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_clip"
    CATEGORY = "Bmad/image"

    @classmethod
    @abstractmethod
    def INPUT_TYPES(s):
        pass

    def clip(self, image, clip_color_255RGB, target, complement):
        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = np.array(image)

        complement_color = [0, 0, 0] if complement == "TO_BLACK" else [255, 255, 255]
        target_color = [0, 0, 0] if target == "TO_BLACK" else [255, 255, 255]

        # if complement is the same as clip color, then TO_**** in target will result in an empty canvas
        # such behavior might leave users confused.
        # by adding an extra step, the expected output is obtained in such cases
        extra_complement_step = tuple(complement_color) == clip_color_255RGB
        first_complement_color = complement_color if not extra_complement_step else [32, 32, 32]

        if complement != "NOTHING":
            image[np.any(image != clip_color_255RGB, axis=-1)] = first_complement_color
        if target != "NOTHING":
            image[np.all(image == clip_color_255RGB, axis=-1)] = target_color
        if extra_complement_step:
            image[np.all(image == first_complement_color, axis=-1)] = complement_color

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return image


image_output_formats_options_map = {
    "RGB": 3,
    "GRAY": 1
}
image_output_formats_options = list(image_output_formats_options_map.keys())


def tensor2opencv(image_tensor, out_format_number_of_channels=3):
    """
    Args:
        image_tensor: tensor containing the image data.
        out_format_number_of_channels: 3 for 'RGB' (default); 4 for 'RGBA' ; 1 for 'GRAY'
    Returns: Numpy int8 array with a RGB24 encoded image
    """
    accepted_out_formats = [1, 3, 4]
    if not out_format_number_of_channels in accepted_out_formats:
        raise ValueError(f"out_format_number_of_channels = {out_format_number_of_channels}, must be one of the "
                         f"following values: {accepted_out_formats}")

    img = np.clip(255. * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    in_format_NoC = 1 if len(list(image_tensor.size())) == 3 else image_tensor.size(dim=3)
    img = maybe_convert_img(img, in_format_NoC, out_format_number_of_channels)

    return img


def maybe_convert_img(img, src_format_number_of_channels, dst_format_number_of_channels):
    """
    Auxiliary method to convert images between the formats: RGB24 ; GRAY8 ; and RGBA32.
    If the number of channels of both formats is the same, the original img is returned unchanged.
    Args:
        img: numpy int8 array with the image
        dst_format_number_of_channels: number of channels of img 
        src_format_number_of_channels: target number of channels 
    Returns:
        Image in the target format (RGB24, GRAY8 or RGBA32).
    """
    import cv2 as cv
    if dst_format_number_of_channels == src_format_number_of_channels:
        return img
    if dst_format_number_of_channels == 3:
        match src_format_number_of_channels:
            case 1:
                return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            case 4:
                return cv.cvtColor(img, cv.COLOR_RGBA2RGB)
    if dst_format_number_of_channels == 1:
        match src_format_number_of_channels:
            case 3:
                return cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            case 4:
                return cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
    if dst_format_number_of_channels == 4:
        match src_format_number_of_channels:
            case 1:
                return cv.cvtColor(img, cv.COLOR_GRAY2RGBA)
            case 3:
                return cv.cvtColor(img, cv.COLOR_RGB2RGBA)
    print("Case not considered for given number of channels: "
          f"source={dst_format_number_of_channels} and target={src_format_number_of_channels}.")
    return None


def opencv2tensor(image):
    """ supposes the image is stored as an int8 numpy array; does not check for the image format """
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


def cache_with_ids(single: bool = False):
    def decorator(func):
        cache = {}

        def wrapper(*args, **kwargs):
            id_args = tuple(map(id, args))
            if id_args in cache:
                # cache hit
                return cache[id_args]
            else:
                # cache miss
                if single:  # only keep the last cached
                    cache.clear()
                result = func(*args, **kwargs)
                cache[id_args] = result
                return result

        def clear_cache():
            cache.clear()

        wrapper.cache_clear = clear_cache  # Attach the clear_cache function

        return wrapper

    return decorator


def prepare_text_for_eval(text, complementary_purge_list=None):
    import re

    # purge the string from domonic entities
    for item in ["exec", "import", "eval", "lambda", "_name_", "_class_", "_bases_",
                 "write", "save", "store", "read", "open", "load", "from", "file"]:
        text = text.replace(item, "")

    if complementary_purge_list is not None:
        for item in complementary_purge_list:
            text = text.replace(item, "")

    # remove comments and new lines
    text = re.sub('#.+', '', text)
    text = re.sub('\n', '', text)

    return text


rect_modes_map = {
        'top-left XY + WH': {
            "toBounds": lambda x1, y1, w, h: (x1, y1, x1 + w, y1 + w),
            "fromBounds": lambda x1, y1, x2, y2: (x1, y1, x2 - x1, y2 - y1)
        },
        'top-left XY + bottom-right XY': {
            # do nothing "
            "toBounds": lambda x1, y1, x2, y2: (x1, y1, x2, y2),
            "fromBounds": lambda x1, y1, x2, y2: (x1, y1, x2, y2)
        },
        'center XY (floored) + WH': {
            "toBounds": lambda x, y, w, h: (x - w // 2 + w & 1, y - h // 2 + h & 1, x + w // 2, y + h // 2),
            "fromBounds": lambda x1, y1, x2, y2: ((x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1)
        },
        'center XY + half WH (all floored)': {
            # can't guarantee original bounds if converted previously, will only have even numbers on width/height
            "toBounds": lambda x, y, half_w, half_h: (x - half_w + 1, y - half_h + 1, x + half_w + 1, y + half_h + 1),
            "fromBounds": lambda x1, y1, x2, y2: ((x1 + x2) // 2, (y1 + y2) // 2, (x2 - x1) // 2, (y2 - y1) // 2)
        }
    }
rect_modes = list(rect_modes_map.keys())
