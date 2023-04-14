import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from io import BytesIO
import base64
import hashlib
import json
import os
import sys
import folder_paths



class CreateRequestMetadata:
    """
    Creates a json file with information pertaining to the prompt request.
    Also implements static methods to access and modify this json.
    There should only be ONE instance of this node in a prompt.
    """
    
    request_id = ""
    output_dir = ""

    def __init__(self):
        self.type = "output"
        print("CREATED")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "request_id": ("STRING", {"default": "insert request id here"})
                     },
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ()
    FUNCTION = "update_outdata"
    CATEGORY = "Bmad/api"
    OUTPUT_NODE = True
    
    
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
        try:
            filename = CreateRequestMetadata.get_request_status_file_path()
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

        # TODO (remove note or change solution) 
        # note: regarding resources w/ same name
        # right now keeps existing data and just issues some warning. 
        # r.n. I feel like is too soon to start placing restrictions without 
        # having a somewhat complete solution/workflow, but this behavior may mislead potential users.
        resource_already_registered = False
        for item in data["outputs"]:
            if item["name"] == resource_name:
                print(f"\033[93m'{resource_name}' resource is already registered in: '{filename}'\033[00m")
                print(f"\033[93mThe new '{resource_name}' register attempt will be ignored.\033[00m")
                resource_already_registered = True
                break
        
        if not resource_already_registered:
            data["outputs"].append({"name":resource_name, "resource":resource_filename})
        
        with open(filename, 'w') as f:
            json.dump(data, f)


    @staticmethod
    def update_request_state(state):
        try:
            filename = CreateRequestMetadata.get_request_status_file_path()
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"\033[91mMetadata file not found: {filename}\033[00m")
            return

        data["state"] = "complete"
        with open(filename, 'w') as f:
            json.dump(data, f)

    
    def update_outdata(self, request_id, extra_pnginfo, unique_id):
        def request_id_is_unique():
            for node in extra_pnginfo["workflow"]["nodes"]:
                if node["type"] != int(unique_id) and node["type"] == "CreateRequestMetadata":
                    return True
            return False
        
        if not request_id_is_unique():
            raise("More than one request node found.")
        
        # no problems found, set the request id
        CreateRequestMetadata.request_id = request_id
        CreateRequestMetadata.output_dir = folder_paths.get_output_directory()

        # get output path
        filename = CreateRequestMetadata.get_request_status_file_path()
        
        # write request status to json file
        request_info = {"state": "started", "outputs":[]}
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
    def INPUT_TYPES(s):
        return {"required": {
                     "resource_0": ("TASK_DONE", )
                     },
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ()
    FUNCTION = "update_outdata"
    CATEGORY = "Bmad/api"
    OUTPUT_NODE = True
    

    def update_outdata(self, extra_pnginfo, unique_id, **kwargs):
        CreateRequestMetadata.update_request_state("complete")
        
        # TODO
        # Validate received tasks with all the info in the outputs
        # if they do not match, add some additional info to inform something went wrong
        # then, update class description w/ this detail

        return ()
        


class SaveImage2:
    """
    Saves image without storing any metadata using a hashcode.
    Outputs from these nodes should be sent to SetRequestStateToComplete.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "resource_name": ("STRING", {"default": "image"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("TASK_DONE", )
    FUNCTION = "save_images"

    CATEGORY = "Bmad/image"
    

    def save_images(self, images, resource_name="image", prompt=None, extra_pnginfo=None):
        def build_hashcode(data):
            if isinstance(data, str):
                data = data.encode(encoding = 'UTF-8', errors = 'strict')
            hash_object = hashlib.sha256() 
            hash_object.update(data)
            return hash_object.hexdigest()
        
        hexdigest = build_hashcode(json.dumps(prompt) + resource_name)
                
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
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            #if save_meta_data: TODO add an option for this
            #   if prompt is not None:
            #       metadata.add_text("prompt", json.dumps(prompt))
            #   if extra_pnginfo is not None:
            #       for x in extra_pnginfo:
            #           metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            });
            counter += 1
        
        # TODO
        # forgot about precessing batches...
        # should add a resource for each image created?
        # or an additional field to indicate a range of items?
        # to be decided later

        CreateRequestMetadata.add_resource(resource_name, hexdigest)
        return (resource_name, )#{ "ui": { "images": results } }
        

class LoadImage64:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {
                     "image_code": ("STRING", {"default": "insert encoded image here"})
                    },
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "get_image"

    CATEGORY = "Bmad/image"
    

    def get_image(self, image_code):
        image = Image.open(BytesIO(base64.b64decode(image_code)))
        image = image.convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image, )
        


NODE_CLASS_MAPPINGS = {
    "CreateRequestMetadata": CreateRequestMetadata,
    "SetRequestStateToComplete": SetRequestStateToComplete, 
    "Save Image 2 ( ! )": SaveImage2,
    "LoadImage64":LoadImage64
}
