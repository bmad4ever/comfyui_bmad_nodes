# utils for color related stuff
from functools import lru_cache, cached_property

import torch
import numpy as np
from PIL import ImageColor, Image
from .dry import print_yellow, circ_quantiles, pseudo_circ_median, circular_stdev, circular_mean

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
    def get_types(cls, advanced=False):
        types = {"required": {"image": ("IMAGE",)}}
        operations = cls.OPERATION[2:] if advanced else cls.OPERATION[0:3]
        types["required"]["target"] = (operations, {"default": operations[0]})
        types["required"]["complement"] = (operations, {"default": operations[1]})
        types["required"]["color"] = ("COLOR",)

        if advanced:
            types["optional"] = {
                "color_a": ("COLOR",),
                "color_b": ("COLOR",)
            }

        return types

    def clip(self, image, color, target, complement, color_a=None, color_b=None, leeway: int = 0):
        """
        @param leeway: per channel, difference threshold
        @return:
        """

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

        if leeway == 0:
            complement_indexes = np.any(image != color, axis=-1)
        else:
            image = image.astype(np.int16)
            leeway = np.array([leeway]*3).astype(np.int16)
            complement_indexes = np.any(np.abs(image - color) > leeway, axis=-1)

        match complement:
            case "NOTHING":
                new_image[complement_indexes] = image[complement_indexes]
            case _:
                new_image[complement_indexes] = complement_color

        image = np.array(new_image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return image


def compute_normalized_inverse_squared_difference_matrix(image, target_color, power=2):
    """
    Compute an inverse squared difference matrix representing the inverse squared euclidean distance of each pixel in
    the image to the target color.

    Args:
        image (numpy.ndarray): The input image (RGB format).
        target_color (tuple): The target color in RGB format.
        power (float): Instead of using the squared difference, use the given power instead.
    Returns:
        numpy.ndarray: A matrix where each element is the normalized inverse squared Euclidean distance between the pixel color
                        (technically not squared unless power = 2)
        and the target color.
    """
    max_distance = (255 ** 2 * 3) ** (1 / 2)
    squared_diff_matrix = np.sqrt(np.sum(abs(image - target_color) ** 2, axis=2))
    inverse_squared_diff_matrix = 2 / (1 + (squared_diff_matrix / max_distance) ** power) - 1
    return inverse_squared_diff_matrix


def compute_average_nisd(image, target_color, mask=None, power=2):
    """
    @param image: image to analyse
    @param target_color: color to compute the "distance" to
    @param mask: tuple: normalized mask; number of zeroes
    @param power: adjust concavity of the "distance weight" function
    @return: the fitness of the color;
             how well it complements the source image within the given mask;
             lower values are better.
    """
    # difference matrix with the euclidean distances
    diff_matrix = compute_normalized_inverse_squared_difference_matrix(image, target_color, power)

    # return mean or apply the mask if provided
    if mask is None:
        return np.mean(diff_matrix)
    else:
        masked_pixel_count = mask[1]
        mask = mask[0]

    if masked_pixel_count == 0:
        return 0.0  # handle the case where there are no masked pixels

    # calculate the average distance for masked pixels
    masked_diff_matrix = mask[:, :] * diff_matrix
    average_nisd = np.sum(masked_diff_matrix) / masked_pixel_count
    return average_nisd


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

    if mask is not None:
        mask = mask[:, :] / 255
        non_zero_count = np.count_nonzero(mask)  # compute only once and wrap within sent arg
        mask = (mask, non_zero_count)

    for color_name, color_rgb in color_dict.items():
        avg_isd = compute_average_nisd(image, color_rgb, mask, power)

        if avg_isd < lowest_avg_distance:
            lowest_avg_distance = avg_isd
            closest_color = color_name

    return closest_color


class Interval:
    def __init__(self, value: list):
        self.value = [value[0], value[-1]]

    def __getitem__(self, key):
        return self.value[key]

    def scale_by_factor(self, scale_factor, center=0):
        bounds = [self.value[0], self.value[1]]

        if center != 0:
            if center < self.value[0] or center > self.value[1]:
                raise ValueError(f"Invalid center. Center must be contained within lower and upper bounds:"
                                 f" [{bounds}], but got -> {center}")
            left_interval = center - bounds[0]
            right_interval = bounds[1] - center
            return Interval([center - left_interval * scale_factor, center + right_interval * scale_factor])

        half_interval = (bounds[1] - bounds[0]) / 2
        bounds = [bounds[0] + half_interval, bounds[1] - half_interval]  # set bounds to center
        half_new_interval = half_interval * scale_factor
        return Interval([bounds[0] - half_new_interval, bounds[1] + half_new_interval])

    def scale_by_constant(self, units, units2=0):
        bounds = [self.value[0], self.value[1]]
        bounds[0] -= units / 2 if units2 == 0 else units
        bounds[1] += units / 2 if units2 == 0 else units2
        return Interval(bounds)

    def interpolate(self, weight: float, other_interval: list):
        """
        Args:
            weight: [0, 1] 0 = original interval; 1 = other_interval
            other_interval:
        """
        wc = 1 - weight
        bounds = [self.value[0] * wc + other_interval[0] * weight, self.value[1] * wc + other_interval[1] * weight]
        return Interval(bounds)

    def clamp(self, lower, upper):
        bounds = HSV_Samples.maxmin([self.value, HSV_Samples.to_interval(lower, upper)]).value
        return Interval(bounds)


class HSV_Samples:
    """
    Stores HSV samples and caches results from computations done over these samples.
    """

    def __init__(self, samples):
        self.samples = samples

    @staticmethod
    def rad2hue(rad: float) -> float:
        """
        Note: won't round the result.
        Roundings are done only over the final intervals to avoid precision errors.
        """
        return np.rad2deg(rad) / 2

    @cached_property
    def hue_avg_rads(self):
        return circular_mean(self.hues_rads)

    @cached_property
    def hues_rads(self):
        hues_degs = self.samples[:, 0].astype(float) * 2  # turn to 360 degrees
        hues_rads = [np.deg2rad(h) for h in hues_degs]
        return hues_rads

    @cached_property
    def h_std_dev(self):
        hue_circ_stddev = circular_stdev(self.hues_rads)
        return self.rad2hue(hue_circ_stddev)

    @cached_property
    def s_std_dev(self):
        from statistics import stdev
        return stdev(self.samples[:, 1], self.s_avg)

    @cached_property
    def v_std_dev(self):
        from statistics import stdev
        return stdev(self.samples[:, 2], self.v_avg)

    @cached_property
    def h_max_dev(self):
        from math import pi
        max_deviation_hue = max([min(abs(pi * 2 - abs(self.hue_avg_rads - a)),
                                     abs(self.hue_avg_rads - a))
                                 for a in self.hues_rads])
        return self.rad2hue(max_deviation_hue)

    @cached_property
    def s_max_dev(self):
        return np.max(np.abs(self.samples[:, 1] - self.s_avg))

    @cached_property
    def v_max_dev(self):
        return np.max(np.abs(self.samples[:, 2] - self.v_avg))

    @cached_property
    def h_mode(self):
        from scipy import stats as st
        return st.mode(self.samples[:, 0])[0]

    @cached_property
    def s_mode(self):
        from scipy import stats as st
        return st.mode(self.samples[:, 1])[0]

    @cached_property
    def v_mode(self):
        from scipy import stats as st
        return st.mode(self.samples[:, 2])[0]

    @cached_property
    def h_pseudo_median(self):
        return pseudo_circ_median(self.h_avg, self.h_mode, 180)

    @cached_property
    def s_median(self):
        return np.median(self.samples[:, 1])

    @cached_property
    def v_median(self):
        return np.median(self.samples[:, 2])

    @cached_property
    def h_avg(self):
        return self.rad2hue(self.hue_avg_rads)

    @cached_property
    def s_avg(self):
        return np.mean(self.samples[:, 1])

    @cached_property
    def v_avg(self):
        return np.mean(self.samples[:, 2])

    @lru_cache(maxsize=2)
    def s_quant(self, quantile):
        return np.quantile(self.samples[:, 1], quantile)

    @lru_cache(maxsize=2)
    def s_quant2(self, lower, upper):
        return Interval([self.s_quant(lower), self.s_quant(upper)])

    @lru_cache(maxsize=2)
    def v_quant(self, quantile):
        return np.quantile(self.samples[:, 2], quantile)

    @lru_cache(maxsize=2)
    def v_quant2(self, lower, upper):
        return Interval([self.v_quant(lower), self.v_quant(upper)])

    @cached_property
    def h_median(self):  # prob. the best guess for most cases
        center_rads = np.deg2rad(self.h_pseudo_median * 2)
        value = circ_quantiles(self.hues_rads, center_rads, [0.5])[0]
        return self.rad2hue(value)

    @lru_cache(maxsize=2)
    def h_quant(self, quantile):
        """
        Args:
            quantile: [0, 1]
        Returns:
        """
        center_rads = np.deg2rad(self.h_median * 2)
        value = circ_quantiles(self.hues_rads, center_rads, [quantile])[0]
        value = self.rad2hue(value)

        # unfix values (fix in the last step, after all changes are made to the interval)
        if quantile < 0.5 and value > self.h_median:
            value -= 180
        if quantile > 0.5 and value < self.h_median:
            value += 180

        return value

    @lru_cache(maxsize=2)
    def h_quant2(self, lower, upper) -> Interval:
        if lower > 0.5 or upper < 0.5:
            raise ValueError("Arguments outside of expected range"
                             "\nexpected: lower <= 0.5 <= higher"
                             f"\ngot: lower={lower} and higher={upper}")

        center_rads = np.deg2rad(self.h_median * 2)
        bounds = circ_quantiles(self.hues_rads, center_rads, [lower, upper])
        bounds = [self.rad2hue(q) for q in bounds]

        # unfix values (fix in the last step, after all changes are made to the interval)
        if bounds[0] > self.h_median:
            bounds[0] -= 180
        if bounds[1] < self.h_median:
            bounds[1] += 180

        return Interval(bounds)

    @staticmethod
    def to_interval(lower, upper):
        return Interval([lower, upper])

    @staticmethod
    def minmax(intervals: list[Interval]):
        return Interval([
            min(lower[0] for lower in intervals),
            max(upper[1] for upper in intervals)
        ])

    @staticmethod
    def maxmin(intervals: list[Interval]):
        return Interval([
            max(lower[0] for lower in intervals),
            min(upper[1] for upper in intervals)
        ])
