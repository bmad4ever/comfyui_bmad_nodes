#from gray2color.utils.vistas import *
from gray2color.utils.ade20k import *
from .color_utils import ColorClip


# note: first pallet item is the black color, which is not relative to any label, and must be added or skipped
ade20k_class_names[:0] = ['None']
ADE20K_dic = {ade20k_class_names[i]: pallet_ade20k[0][i] for i in range(len(ade20k_class_names))}

class ColorClipADE20K(ColorClip):
    @classmethod
    def INPUT_TYPES(s):
        types = super().get_types(advanced=False)
        types["required"].pop("color")
        types["required"]["class_name"] = (ade20k_class_names, {"default": 'animal, animate being, beast, brute, creature, fauna'})
        return types

    def color_clip(self, image, class_name, target, complement):
        clip_color = list((ADE20K_dic[class_name]*255).astype(np.uint8))
        image = self.clip(image, clip_color, target, complement)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "Color Clip ADE20k": ColorClipADE20K
}
