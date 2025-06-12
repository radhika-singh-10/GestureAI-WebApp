# loader.py
import torch
from model import Gesture3DCNN
from collections import OrderedDict

def load_model(checkpoint_path):
    model = Gesture3DCNN()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        print(f"{k} : {tuple(v.shape)}")
        new_key = k.replace('module.', '') 
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model




