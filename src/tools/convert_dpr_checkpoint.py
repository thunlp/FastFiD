import torch
import os
from collections import OrderedDict

# upgrade version of DPR
# Top-k passages	Original DPR NQ model	New DPR model
#       1                   45.87                  52.47
#       5                   68.14                  72.24
#      20                   79.97                  81.33
#     100                   85.87                  87.29

# state_dict = torch.load("/data1/private/huangyufei/pretrained_models/bmtrain/t5-base/pytorch_model.pt")
# state_dict = torch.load("checkpoints/generative_top100_constantlr/epoch0/pytorch_model.pt")
state_dict = torch.load("/data1/private/huangyufei/pretrained_models/DenseRetrieval/hf_bert_base.cp", map_location='cpu')
output_dir = "/data1/private/huangyufei/pretrained_models/DenseRetrieval/dpr-{}_encoder-ver2"
# state_dict_2 = torch.load("/data1/private/huangyufei/pretrained_models/DenseRetrieval/dpr-question_encoder-single-nq-base/pytorch_model.bin", map_location='cpu')
ctx_model = OrderedDict()
question_model = OrderedDict()
for key in state_dict['model_dict']:
    if key.startswith("ctx_model"):
        new_key =  'ctx_encoder.bert_model' + key[len("ctx_model"):]
        ctx_model[new_key] = state_dict['model_dict'][key]
    else:
        new_key = 'question_encoder.bert_model' + key[len("question_model"):]
        question_model[new_key] = state_dict['model_dict'][key]
        # print(key)
ctx_model_output_dir = output_dir.format('ctx')
os.makedirs(ctx_model_output_dir, exist_ok=True)
torch.save(ctx_model, os.path.join(ctx_model_output_dir, 'pytorch_model.bin'))
question_model_output_dir = output_dir.format('question')
os.makedirs(question_model_output_dir, exist_ok=True)
torch.save(question_model, os.path.join(question_model_output_dir, 'pytorch_model.bin'))
