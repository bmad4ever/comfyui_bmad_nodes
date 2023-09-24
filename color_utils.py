# utils for color related stuff

import torch
import numpy as np
from PIL import ImageColor, Image
from .dry import print_yellow

color255_INPUT = ("INT", {
    "default": 0,
    "min": 0,
    "max": 255,
    "step": 1
})


def setup_color_to_correct_type(color):
    if color is None:
        return None
    return color if isinstance(color, list) else ImageColor.getcolor(color, "RGB")


class ColorClip:
    OPERATION = [
        "TO_BLACK",
        "TO_WHITE",
        "NOTHING",
        "TO_A",
        "TO_B"
    ]

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_clip"
    CATEGORY = "Bmad/image"

    @classmethod
    def get_types(self, advanced=False):
        types = {"required": {"image": ("IMAGE",)}}
        operations = self.OPERATION[2:] if advanced else self.OPERATION[0:3]
        types["required"]["target"] = (operations, {"default": operations[0]})
        types["required"]["complement"] = (operations, {"default": operations[1]})
        types["required"]["color"] = ("COLOR",)

        if advanced:
            types["optional"] = {
                "color_a": ("COLOR",),
                "color_b": ("COLOR",)
            }

        return types

    def clip(self, image, color, target, complement, color_a=None, color_b=None):

        color = setup_color_to_correct_type(color)
        color_a = setup_color_to_correct_type(color_a)
        color_b = setup_color_to_correct_type(color_b)

        image = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image = np.array(image)

        def select_color(selection):
            match selection:
                case "TO_BLACK":
                    return [0, 0, 0]
                case "TO_WHITE":
                    return [255, 255, 255]
                case "TO_A":
                    if color_a is None:
                        print_yellow("color_a was not set")
                    return color_a
                case "TO_B":
                    if color_b is None:
                        print_yellow("color_b was not set")
                    return color_b
                case "_":
                    return None

        complement_color = select_color(complement)
        target_color = select_color(target)

        match target:
            case "NOTHING":
                new_image = np.array(image, copy=True)
            case _:
                new_image = np.full_like(image, target_color)

        complement_indexes = np.any(image != color, axis=-1)
        match complement:
            case "NOTHING":
                new_image[complement_indexes] = image[complement_indexes]
            case _:
                new_image[complement_indexes] = complement_color

        image = np.array(new_image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return image


def compute_inverse_squared_difference_matrix(image, target_color, power=2):
    """
    Compute an inverse squared difference matrix representing the inverse squared euclidean distance of each pixel in
    the image to the target color.

    Args:
        image (numpy.ndarray): The input image (RGB format).
        target_color (tuple): The target color in RGB format.
        power (float): Instead of using the squared difference, use the given power instead.
    Returns:
        numpy.ndarray: A matrix where each element is the inverse squared Euclidean distance between the pixel color
        and the target color.
    """
    squared_diff_matrix = np.sum(abs(image - target_color) ** power, axis=2)
    inverse_squared_diff_matrix = 1 / (1 + squared_diff_matrix)
    return inverse_squared_diff_matrix


def compute_average_isd(image, target_color, mask=None, power=2):
    # difference matrix with the euclidean distances
    diff_matrix = compute_inverse_squared_difference_matrix(image, target_color, power)

    # return mean or apply the mask if provided
    if mask is None:
        return np.mean(diff_matrix)
    masked_diff_matrix = np.where(mask[:, :, np.newaxis], diff_matrix, 0)

    # calculate the average distance for masked pixels
    masked_pixel_count = np.count_nonzero(mask)
    if masked_pixel_count == 0:
        return 0.0  # handle the case where there are no masked pixels

    average_isd = np.sum(masked_diff_matrix) / masked_pixel_count
    return average_isd


def find_complementary_color(image, color_dict, mask=None, power=2):
    """
    Selects the color from the given dictionary that has the least similar colors present in the image by
    finding the color with the lowest ISD average; where isd is the inverse squared difference from a pixel to the color.

    ISD of a pixel is given by: 1/(1+d**2), where d is the euclidean distance of a pixel to the target color.
    The closer the colors are, the more they weight in the final average.

    Args:
        image: The src image used to compute the complement.
        color_dict: Dictionary whose keys should be the names of the colors and the values RGB coded color as a tuple.
        mask: Optional mask that indicates the target pixels. The distance to pixels outside the mask is disregarded.
        power: Override power of 2 in ISD, will use this instead of squaring.
    Returns:
        The complement color name, as defined in color_dict.
    """
    lowest_avg_distance = 2  # 1 is the theoretical maximum possible value
    closest_color = None

    for color_name, color_rgb in color_dict.items():
        avg_isd = compute_average_isd(image, color_rgb, mask, power)

        if avg_isd < lowest_avg_distance:
            lowest_avg_distance = avg_isd
            closest_color = color_name

    return closest_color
