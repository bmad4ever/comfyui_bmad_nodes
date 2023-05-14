#import traceback

loaded = []
not_loaded = []
exceptions = []


try:
    from .api_nodes import NODE_CLASS_MAPPINGS as api
    loaded.append("api nodes")
except Exception as e:
    repeat = {}
    not_loaded.append("api nodes")
    exceptions.append(e)

try:
    from .simple_utilities import NODE_CLASS_MAPPINGS as simple
    loaded.append("simple utility nodes")
except Exception as e:
    cc_ade20k = {}
    exceptions.append(e)
    not_loaded.append("simple utility nodes")

try:
    from .grabcut_nodes import NODE_CLASS_MAPPINGS as grab
    loaded.append("grab cut nodes")
except Exception as e:
    grab = {}
    not_loaded.append("grab cut nodes")
    exceptions.append(e)

try:
    from .color_clip_ade20k import NODE_CLASS_MAPPINGS as cc_ade20k
    loaded.append("color clip ade20k node")
except Exception as e:
    cc_ade20k = {}
    exceptions.append(e)
    not_loaded.append("color clip ade20k node")

try:
    from .otsu_threshold_node import NODE_CLASS_MAPPINGS as otsu
    loaded.append("otsu filter node")
except Exception as e:
    otsu = {}
    not_loaded.append("otsu filter node")
    exceptions.append(e)


import __main__
import os

NODE_CLASS_MAPPINGS = {**api, **simple, **grab, **cc_ade20k, **otsu}

__all__ = ['NODE_CLASS_MAPPINGS']

extentions_folder = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)), "web\extensions\Bmad")



print('\n\033[1mFrom Bmad Custom Nodes\033[0m')

if not os.path.exists(extentions_folder):
    exceptions.append('"web\extensions\Bmad" folder is missing, some nodes won\'t work as intended.')


print(f' \033[92mLoaded:')
for m in loaded:
    print(f'  + {m}')

if len(not_loaded) > 0:
    print(f' \033[93mNot loaded:')
    for m in not_loaded:
        print(f'  * {m}')

if len(exceptions) > 0:
    print(' \033[91mProblems:')
    for e in exceptions:
        print(f'  ! {e}')

print('\033[0m')