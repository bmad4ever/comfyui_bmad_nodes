# D.R.Y ( Don't Repeat Yourself: cross file utilities )

from abc import abstractmethod
import torch
import numpy as np
from PIL import Image, ImageOps


color255Input = ("INT", {
    "default": 0,
    "min": 0,
    "max": 255,
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

        if complement != "NOTHING":
            complement_color = [0, 0, 0] if complement == "TO_BLACK" else [255, 255, 255]
            image[np.any(image != clip_color_255RGB, axis=-1)] = complement_color
        if target != "NOTHING":
            target_color = [0, 0, 0] if target == "TO_BLACK" else [255, 255, 255]
            image[np.all(image == clip_color_255RGB, axis=-1)] = target_color

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return image
