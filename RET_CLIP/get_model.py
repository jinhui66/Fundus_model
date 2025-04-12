
import torch
import torch.nn as nn
import json
from RET_CLIP.clip.model import CLIP

def get_retclip_model():
    # Build the CLIP model
    vision_model_config_file = f"RET_CLIP/clip/model_configs/ViT-B-16.json"
    print('Loading vision model config from', vision_model_config_file)

    text_model_config_file = "RET_CLIP/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json"
    print('Loading text model config from', text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])
        for k, v in json.load(ft).items():
            model_info[k] = v
    model_info['use_flash_attention'] = False

    model = CLIP(**model_info)
    pretrained_dict = torch.load("./RET_CLIP/ret-clip.pt", map_location=torch.device('cpu'))
    from collections import OrderedDict
    
    # 移除 "module." 前缀
    pretrained_dict = OrderedDict({
        k.replace("module.", ""): v 
        for k, v in pretrained_dict.items()
    })
    model.load_state_dict(pretrained_dict)
    # print("输出模型:", model)
    return model