# D.R.Y ( Don't Repeat Yourself: cross file utilities )
from typing import Callable
import torch
import numpy as np

base_category_path = "Bmad"
images_category_path = f"{base_category_path}/image"
conditioning_category_path = f"{base_category_path}/conditioning"


grid_len_INPUT = ("INT", {
    "default": 3,
    "min": 1,
    "max": 8,
    "step": 1
})

image_output_formats_options_map = {
    "RGB": 3,
    "GRAY": 1
}
image_output_formats_options = list(image_output_formats_options_map.keys())


def tensor2opencv(image_tensor, out_format_number_of_channels=3) -> np.ndarray:
    """
    Args:
        image_tensor: tensor containing the image data.
        out_format_number_of_channels: 3 for 'RGB' (default); 4 for 'RGBA' ; 1 for 'GRAY';
            or 0 for the same number of channels as image_tensor
    Returns: Numpy int8 array with a RGB24 encoded image
    """
    in_format_NoC = 1 if len(list(image_tensor.size())) == 3 else image_tensor.size(dim=3)
    if out_format_number_of_channels == 0:
        out_format_number_of_channels = in_format_NoC

    accepted_out_formats = [1, 3, 4]
    if not out_format_number_of_channels in accepted_out_formats:
        raise ValueError(f"out_format_number_of_channels = {out_format_number_of_channels}, must be one of the "
                         f"following values: {accepted_out_formats}")

    img = np.clip(255. * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
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

# region prepare strings for eval

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

def filter_expression_names(accept_filter_func: Callable[[str], bool], expression: str) -> list[str]:
    """
    Filters name tokens from the given expression.
    Usage example: filter properties so that only the ones in the expression are added to the eval scope.
    """
    import io
    import tokenize
    tokens = tokenize.tokenize(io.BytesIO(expression.encode('utf-8')).readline)
    relevant_names_in_expression = []
    for token_type, token_value, _, _, _ in tokens:
        if token_type == tokenize.NAME and token_value and accept_filter_func(token_value):
            relevant_names_in_expression.append(token_value)
    return relevant_names_in_expression

# endregion

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
            "toBounds": lambda x, y, w, h: (x - w // 2 , y - h // 2, x + w // 2 + (w & 1), y + h // 2 + (h & 1)),
            "fromBounds": lambda x1, y1, x2, y2: ((x1 + x2) // 2, (y1 + y2) // 2, x2 - x1, y2 - y1)
        },
        'center XY + half WH (all floored)': {
            # can't guarantee original bounds if converted previously, will only have even numbers on width/height
            "toBounds": lambda x, y, half_w, half_h: (x - half_w, y - half_h, x + half_w + (half_w == 0), y + half_h + (half_h == 0) ),
            "fromBounds": lambda x1, y1, x2, y2: ((x1 + x2) // 2, (y1 + y2) // 2, (x2 - x1) // 2, (y2 - y1) // 2)
        }
    }
rect_modes = list(rect_modes_map.keys())


# region handle nodes w/ arbitrary number of inputs

def get_arg_name_from_multiple_inputs(type, index):
    """
    Args:
        type: must be lower case
        index: starting at 0, as you would normally use in a list
    Returns:
        The name of the arg setup by ArbitraryInputsFixedOutputs.js
    """
    return f"{type}_{index+1}"


def print_yellow(message):
    print(f"\033[93m{message}\033[0m")


def convert_list_args_to_args(node_name, **kwargs):
    arg_list = []
    for key, value in kwargs.items():
        if len(value) > 1:
            print_yellow(f"{node_name} node is not able to handle '{key}' lists; only the first elem will be used.")
        value = value[0]
        arg_list.append(value)
    return tuple(arg_list)

# endregion


#region math utils

def circular_mean(angles_in_rads):
    """
    Adapted from https://en.wikipedia.org/wiki/Circular_mean.
    Args:
        angles_in_rads: exactly what the name implies
    Returns:
        the circular mean in radians
    """
    import math
    mean_cos = sum(math.cos(angle) for angle in angles_in_rads) / len(angles_in_rads)
    mean_sin = sum(math.sin(angle) for angle in angles_in_rads) / len(angles_in_rads)
    mean_angle = math.atan2(mean_sin, mean_cos)
    if mean_angle < 0:
        mean_angle += math.pi*2
    return mean_angle


def circular_stdev(angles_in_rads):
    """
    Args:
        angles_in_rads: exactly what the name implies
    Returns:
        the circular standard deviation
    """
    import math
    mean_cos = sum(math.cos(angle) for angle in angles_in_rads) / len(angles_in_rads)
    mean_sin = sum(math.sin(angle) for angle in angles_in_rads) / len(angles_in_rads)
    R = math.sqrt(mean_cos**2 + mean_sin**2)  # ]0, 1] where 0 = dispersed and 1 = concentrated
    if R >= 1:
        return 0
    if R <= 29e-10:
        return math.pi * 2
    std_dev = math.sqrt(-2 * math.log(R))
    return std_dev


def circ_quantiles(samples, center, quantiles):
    """
    Args:
        samples: the sample data in radians.
        center: center defines the point at quantile 0.5.
        quantiles (list): quantiles to fetch.
    Returns:
        The respective values for each quantile.
    """
    tau = 2*np.pi
    offset = np.pi - center
    offset_data = [(s + offset + tau) % tau for s in samples]
    values_at_quantiles = np.quantile(offset_data, quantiles)
    adjusted_values = [(v - offset + tau) % tau for v in values_at_quantiles]

    return adjusted_values


def pseudo_circ_median(mean, mode, interval_size=0):
    """
    Similar computation to: (mean + mode)/2;
     but taking into account the circular nature of the data.
    Args:
        mean: the mean either in radians or within the given interval
        mode: the mode either in radians or within the given interval
        interval_size: the interval in which the source data is contained,
         i.e. specifies the upper bound which coincides with the lower bound due to the circular nature of the data.
    Returns:
        the pseudo median either in radians or within the given interval.
    """
    if interval_size > 0:
        mean = mean / interval_size * 2 * np.pi
        mode = mode / interval_size * 2 * np.pi

    pseudo_median = circular_mean([mean, mode])

    if interval_size > 0:
        pseudo_median = pseudo_median / (2 * np.pi) * interval_size

    return pseudo_median


#endregion
