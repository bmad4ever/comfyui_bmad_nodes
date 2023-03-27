import torch
import numpy as np
from PIL import Image, ImageOps

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

    CATEGORY = "image"

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

NODE_CLASS_MAPPINGS = {
    "MonoMerge": MonoMerge
}