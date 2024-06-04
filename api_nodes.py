import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
import base64
import hashlib
import json
import copy
import os
import folder_paths
import nodes
import comfy_extras.nodes_hypernetwork as hyper
from .utils.dry import base_category_path

api_category_path = f"{base_category_path}/api"

# region : api core nodes


class CreateRequestMetadata:
    """
    Creates a json file with information pertaining to the prompt request.
    Also implements static methods to access and modify this json.
    There should only be ONE instance of this node in a prompt.
    """

    request_id = None
    output_dir = ""

    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "request_id": ("STRING", {"default": "insert_id"})
        },
        }

    RETURN_TYPES = ()
    FUNCTION = "update_outdata"
    CATEGORY = api_category_path
    OUTPUT_NODE = True

    @staticmethod
    def get_and_validate_requestID():
        if CreateRequestMetadata.request_id is None:
            raise TypeError("Request ID was not set. CreateRequestMetadata node might be missing.")
        if CreateRequestMetadata.request_id == "":
            raise ValueError("Request ID was set to empty."
                             " Check if it is being properly set to avoid conflicts with subsequent requests.")
        return CreateRequestMetadata.request_id

    @staticmethod
    def get_request_status_file_name():
        return f"{CreateRequestMetadata.request_id}.json"

    @staticmethod
    def get_request_status_file_path():
        file = CreateRequestMetadata.get_request_status_file_name()
        filename = os.path.join(CreateRequestMetadata.output_dir, file)
        return filename

    @staticmethod
    def get_request_metadata():
        filename = CreateRequestMetadata.get_request_status_file_path()
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"\033[91mMetadata file not found: {filename}\033[00m")
            return None
        return data

    @staticmethod
    def add_resource(resource_name, resource_filename):
        filename = CreateRequestMetadata.get_request_status_file_path()
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"\033[91mMetadata file not found: {filename}\033[00m")
            return

        resource_index = next((index for index, value in enumerate(data["outputs"]) if value["name"] == resource_name),
                              -1)
        resource_already_registered = resource_index != -1

        if not resource_already_registered:
            data["outputs"].append({"name": resource_name, "resource": [resource_filename]})
        else:
            data["outputs"][resource_index]["resource"].append(resource_filename)

        with open(filename, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def add_resource_list(resource_name, resource_filenames):
        filename = CreateRequestMetadata.get_request_status_file_path()
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"\033[91mMetadata file not found: {filename}\033[00m")
            return

        resource_index = next((index for index, value in enumerate(data["outputs"]) if value["name"] == resource_name),
                              -1)
        resource_already_registered = resource_index != -1

        if not resource_already_registered:
            data["outputs"].append({"name": resource_name, "resource": resource_filenames})
        else:
            data["outputs"][resource_index]["resource"].extend(resource_filenames)

        with open(filename, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def update_request_state(state):
        filename = CreateRequestMetadata.get_request_status_file_path()
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"\033[91mMetadata file not found: {filename}\033[00m")
            return

        data["state"] = "complete"
        with open(filename, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def set_error_state(message):
        filename = CreateRequestMetadata.get_request_status_file_path()
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"\033[91mMetadata file not found: {filename}\033[00m")
            return

        data["state"] = "failed"
        data["error"] = message

        with open(filename, 'w') as f:
            json.dump(data, f)

        raise Exception(message)

    def update_outdata(self, request_id):
        if request_id == "insert_id":
            raise ValueError("Request ID in CreateRequestMetadata node with value: "
                             "'insert_id'. You might not be setting it properly or "
                             "might have more than one CreateRequestMetadata node in your workflow/node.")

        assert CreateRequestMetadata.request_id != request_id, (
            "Request ID is equal to previously set ID. "
            "You may have more than one CreateRequestMetadata node in your workflow/prompt.")

        # no problems found, set the request id
        CreateRequestMetadata.request_id = request_id
        CreateRequestMetadata.output_dir = folder_paths.get_output_directory()

        # get output path
        filename = CreateRequestMetadata.get_request_status_file_path()

        # write request status to json file
        request_info = {"state": "started", "outputs": []}
        with open(filename, 'w') as f:
            json.dump(request_info, f)

        return ()


class SetRequestStateToComplete:
    """
    Set request state to 'complete' in the request metadata file.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "resource_0": ("TASK_DONE",)
        },
        }

    RETURN_TYPES = ()
    FUNCTION = "update_outdata"
    CATEGORY = api_category_path
    OUTPUT_NODE = True

    def update_outdata(self, **kwargs):
        # update request file
        CreateRequestMetadata.update_request_state("complete")

        # clear request_id
        CreateRequestMetadata.request_id = None

        # TODO
        # Validate received tasks with all the info in the outputs
        # if they do not match, add some additional info to inform something went wrong
        # then, update class description w/ this detail

        return ()


class SaveImage2:
    """
    Saves image without storing any metadata using a hexdigest as the name.
    Outputs from these nodes should be sent to SetRequestStateToComplete.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"images": ("IMAGE",),
                     "resource_name": ("STRING", {"default": "image"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("TASK_DONE",)
    FUNCTION = "save_images"

    CATEGORY = api_category_path

    def save_images(self, images, resource_name="image", prompt=None, extra_pnginfo=None):
        def build_hashcode(data):
            if isinstance(data, str):
                data = data.encode(encoding='UTF-8', errors='strict')
            hash_object = hashlib.sha256()
            hash_object.update(data)
            return hash_object.hexdigest()

        req_id = CreateRequestMetadata.get_and_validate_requestID()
        hexdigest = build_hashcode(req_id + resource_name)

        def map_filename(filename):
            prefix_len = len(os.path.basename(hexdigest))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)

        subfolder = os.path.dirname(os.path.normpath(hexdigest))
        filename = os.path.basename(os.path.normpath(hexdigest))

        full_output_folder = os.path.join(self.output_dir, subfolder)

        if os.path.commonpath((self.output_dir, os.path.abspath(full_output_folder))) != self.output_dir:
            print("Saving image outside the output folder is not allowed.")
            return {}

        try:
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                                 map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1

        # results = list()
        files = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            # if save_meta_data: TODO add an option for this
            #   if prompt is not None:
            #       metadata.add_text("prompt", json.dumps(prompt))
            #   if extra_pnginfo is not None:
            #       for x in extra_pnginfo:
            #           metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            files.append(file)
            # results.append({
            #    "filename": file,
            #    "subfolder": subfolder,
            #    "type": self.type
            # });
            counter += 1

        CreateRequestMetadata.add_resource_list(resource_name, files)
        return (resource_name,)  # { "ui": { "images": results } }


class LoadImage64:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
                "image_code": ("STRING", {"default": "insert encoded image here"})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_image"
    CATEGORY = api_category_path

    def get_image(self, image_code):
        image = Image.open(BytesIO(base64.b64decode(image_code)))
        image = image.convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)


class RequestInputs:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "values": ("STRING", {"default": ""}),
        },
        }

    RETURN_TYPES = tuple(["STRING" for x in range(32)])
    FUNCTION = "start"
    CATEGORY = api_category_path

    def start(self, values):
        values = tuple(json.loads(values).values())
        return values


# endregion : api core nodes


# region : input converters

class String2Int:
    """
    Under the supposition that this node will receive values from the RequestInputs node,
     will still work with integer values in the json, as int() cast will work both int and str types.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"inStr": ("STRING", {"default": ""})}, }

    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = f"{api_category_path}/parseInput"

    def convert(self, inStr):
        return (int(inStr),)


class String2Float:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"inStr": ("STRING", {"default": ""})}, }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "convert"
    CATEGORY = f"{api_category_path}/parseInput"

    def convert(self, inStr):
        return (float(inStr),)


class InputString2IntArray:
    """
    Under the supposition this will be used with RequestInputs, the integers may already come as an array.
    The input is, therefore, polymorphic and both array and string types are accepted as inputs to both allow a valid
    request json and a mock array given via the web UI.

    When using a string: the integers should be separated with commas
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"inStr": ("STRING", {"default": ""})}, }

    RETURN_TYPES = ("INT_ARRAY",)
    FUNCTION = "convert"
    CATEGORY = f"{api_category_path}/parseInput"

    def convert(self, inStr):
        # not really a str, suppose is a list read from the input json
        if isinstance(inStr, list):
            return (inStr,)

        # otherwise suppose it is a valid string
        return ([int(x) for x in inStr.split(',')],)


# endregion : input converters


# region : dirty loaders

class DirtyLoaderUtils:

    # checks file name without taking into account the file extension;
    # then gets the file with the extension from the list
    @staticmethod
    def find_matching_filename(input_string, filenames):
        input_base, input_ext = os.path.splitext(input_string)
        for filename in filenames:
            filename_base, filename_ext = os.path.splitext(filename)
            if input_base == filename_base:
                return filename  # return matching filename with file extension
        CreateRequestMetadata.set_error_state(f"File '{input_string}' not found.")


class DirtyCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config_name": ("STRING", {"default": ""}),
            "ckpt_name": ("STRING", {"default": ""})
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = f"{api_category_path}/dirty loaders"

    def load_checkpoint(self, config_name, ckpt_name, output_vae=True, output_clip=True):
        ckpt_name = DirtyLoaderUtils.find_matching_filename(
            ckpt_name, folder_paths.get_filename_list("checkpoints"))

        config_name = DirtyLoaderUtils.find_matching_filename(
            config_name, folder_paths.get_filename_list("checkpoints"))

        loader = nodes.CheckpointLoader()
        return loader.load_checkpoint(config_name, ckpt_name, output_vae, output_clip)


class DirtyCheckpointLoaderSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = f"{api_category_path}/dirty loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_name = DirtyLoaderUtils.find_matching_filename(
            ckpt_name, folder_paths.get_filename_list("checkpoints"))

        loader = nodes.CheckpointLoaderSimple()
        return loader.load_checkpoint(ckpt_name, output_vae, output_clip)


class DirtyLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "clip": ("CLIP",),
                             "lora_name": ("STRING", {"default": ""}),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = f"{api_category_path}/dirty loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        lora_name = DirtyLoaderUtils.find_matching_filename(
            lora_name, folder_paths.get_filename_list("loras"))

        loader = nodes.LoraLoader()
        return loader.load_lora(model, clip, lora_name, strength_model, strength_clip)


class DirtyHypernetworkLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "hypernetwork_name": ("STRING", {"default": ""}),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_hypernetwork"

    CATEGORY = f"{api_category_path}/dirty loaders"

    def load_hypernetwork(self, model, hypernetwork_name, strength):
        hypernetwork_name = DirtyLoaderUtils.find_matching_filename(
            hypernetwork_name, folder_paths.get_filename_list("hypernetworks"))

        loader = hyper.HypernetworkLoader()
        return loader.load_hypernetwork(model, hypernetwork_name, strength)


# endregion : dirty loaders


# region : dumpers

class GetModels:
    dump_option = ['all models',
                   'checkpoints',
                   'clip',
                   'clip_vision',
                   'configs',
                   'controlnet',
                   'diffusers',
                   'embeddings',
                   'gligen',
                   'hypernetworks',
                   'loras',
                   'style_models',
                   'upscale_models',
                   'vae'
                   ]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dump": (cls.dump_option, {"default": "all models"})
        }
        }

    RETURN_TYPES = ()
    FUNCTION = "dump_it"
    CATEGORY = f"{base_category_path}/dump"
    OUTPUT_NODE = True

    def dump_it(self, dump):
        dump_data = {}

        if dump == 'all models':
            for item in self.dump_option[1:]:
                dump_data[item] = folder_paths.get_filename_list(item)
        else:
            dump_data['list'] = folder_paths.get_filename_list(dump)

        file = f"{dump}.json"
        file = os.path.join(self.output_dir, file)
        with open(file, 'w') as f:
            json.dump(dump_data, f, indent=1)

        return ()


class GetPrompt:
    prompt_mode = ["print to console", "save to file"]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "api_prompt": (cls.prompt_mode, {"default": cls.prompt_mode[0]})
        },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "getPrompt"
    CATEGORY = f"{base_category_path}/dump"
    OUTPUT_NODE = True

    def getPrompt(self, api_prompt, prompt, unique_id):
        # changing the original will mess the prompt execution, therefore make a copy
        prompt = copy.deepcopy(prompt)

        # remove this node from the prompt
        this_node = prompt[unique_id]
        del prompt[unique_id]

        # remove widgtes inputs from RequestInputs, only "values" is needed.
        for key in prompt:
            if prompt[key]["class_type"] == "RequestInputs":
                inputs = prompt[key]["inputs"]
                for attribute in list(inputs.keys()):
                    if attribute != "values":
                        del inputs[attribute]
                break  # supposes only 1 RequestInputs node exists

        prompt = {"prompt": prompt}

        # print to console or file
        if api_prompt == "print to console":
            print(json.dumps(prompt))
        elif api_prompt == "save to file":
            # TODO
            # avoid collisions (maybe just name it w/ date/time prefix?)
            # instead of owerriding the file
            file = "prompt.json"
            file = os.path.join(self.output_dir, file)
            with open(file, 'w') as f:
                json.dump(prompt, f, indent=1)

        return ()


# endregion : dumpers


NODE_CLASS_MAPPINGS = {
    "CreateRequestMetadata": CreateRequestMetadata,
    "SetRequestStateToComplete": SetRequestStateToComplete,
    "Save Image (api)": SaveImage2,
    "Load 64 Encoded Image": LoadImage64,
    "RequestInputs": RequestInputs,

    "String to Integer": String2Int,
    "String to Float": String2Float,
    "Input/String to Int Array": InputString2IntArray,

    "CheckpointLoader (dirty)": DirtyCheckpointLoader,
    "CheckpointLoaderSimple (dirty)": DirtyCheckpointLoaderSimple,
    "LoraLoader (dirty)": DirtyLoraLoader,
    "HypernetworkLoader (dirty)": DirtyHypernetworkLoader,

    "Get Models": GetModels,
    "Get Prompt": GetPrompt
}
