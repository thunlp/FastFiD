import torch
import json
from collections import OrderedDict
import os

original_path = '/home/huangyufei/data1_hyf/pretrained_models/fid-kd/nq_retriever/'
new_path = '/home/huangyufei/data1_hyf/pretrained_models/fid-kd/nq_retriever_hf/'
state_dict = torch.load(original_path + 'pytorch_model.bin', map_location='cpu')
new_state_dict = OrderedDict()
for key in state_dict:
    new_key = 'bert_model.' + key[6:]
    new_state_dict[new_key] = state_dict[key]
os.makedirs(new_path, exist_ok=True)
torch.save(new_state_dict, new_path + 'pytorch_model.bin')