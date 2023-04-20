from .color_clip_node import NODE_CLASS_MAPPINGS as colorclip
from .grabcut_nodes import NODE_CLASS_MAPPINGS as grab
from .mono_merge_node import NODE_CLASS_MAPPINGS as monomerge
from .otsu_threshold_node import NODE_CLASS_MAPPINGS as otsu
from .repeat_into_grid_node import NODE_CLASS_MAPPINGS as repeat
from .api_nodes import NODE_CLASS_MAPPINGS as api

import __main__
import os

NODE_CLASS_MAPPINGS = {**colorclip, **grab, **monomerge, **otsu, **repeat, **api}

__all__ = ['NODE_CLASS_MAPPINGS']


extentions_folder = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)), "web\extensions\Bmad")

if not os.path.exists(extentions_folder):
    print('"web\extensions\Bmad" folder is missing, some nodes won\'t work as intended.')

print('\033[34mBmad Custom Nodes: \033[92mLoaded\033[0m')