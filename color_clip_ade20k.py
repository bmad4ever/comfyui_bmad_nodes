from gray2color.utils.ade20k import ade20k_class_names, pallet_ade20k
from .utils.color import ColorClip
from numpy import uint8

# note: first pallet item is the black color, which is not relative to any label, and must be added or skipped
ade20k_class_names[:0] = ['None']
ADE20K_dic = {ade20k_class_names[i]: pallet_ade20k[0][i] for i in range(len(ade20k_class_names))}
default_class_name = list(ADE20K_dic.keys())[13]  # person, individual, someone, somebody, mortal, soul


class ColorClipADE20K(ColorClip):
    @classmethod
    def INPUT_TYPES(cls):
        types = super().get_types(advanced=False)
        types["required"].pop("color")
        types["required"]["class_name"] = (ade20k_class_names, {"default": default_class_name})
        return types

    def color_clip(self, image, class_name, target, complement):
        clip_color = list((ADE20K_dic[class_name] * 255).astype(uint8))
        # note: max eucl. dist. between 2 colors in the dictionary is 7.xxx ... w/ a diff of (4, 5, 3)
        image = self.clip(image, clip_color, target, complement, leeway=2)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "Color Clip ADE20k": ColorClipADE20K
}
