from fairseq.data.data_utils import load_indexed_dataset
from fairseq.models.bart import BARTModel
from fairseq.data import Dictionary
import torch.nn.functional as F 

import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os


dataset_name = 'BBBP'
# dataset = dataset_name if dataset_name in set(["BBBP", "BACE", "HIV"]) else f"{dataset_name}_{args.subtask}"
dataset = dataset_name

store_path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data"
model = f"{store_path}/{dataset}/processed/input0"

with open('/home/gayane/BartLM/captum/fairseq/scripts/datasets.json') as f:
    datasets_json = json.load(f)
dataset_js = datasets_json[dataset]
task_type = dataset_js['type']

if task_type == "regression":
    mi = dataset_js['minimum']
    ma = dataset_js['maximum']

os.system(f"mkdir -p {store_path}/{dataset}/")
os.system(f"mkdir -p {store_path}/{dataset}/processed/")
os.system(f"mkdir -p {store_path}/{dataset}/processed/input0/")
os.system(f"mkdir -p {store_path}/{dataset}/processed/label/")

data_type = "valid"

warmup = 163
totNumUpdate = 1020
lr = '3e-5'
chkpt_path = '/mnt/good/gayane/data/chkpt/BBBP_bs_16_dropout_0.1_lr_3e-5_totalNum_1020_warmup_163_noise_type_uniform_r3f_lambda_1.0/checkpoint_best.pt'
# chkpt_path = f"/mnt/good/gayane/data/chkpt/{dataset}_bs_16_lr_{lr}_totalNum_{totNumUpdate}_warmup_{warmup}/checkpoint_last.pt"
print(chkpt_path)  # BACE_bs_16_lr_3e-5_totalNum_1135_warmup_181/ in test 
# bart = BARTModel.from_pretrained(model,  checkpoint_file = chkpt_path, 
bart = BARTModel.from_pretrained(model,  checkpoint_file = '/home/gayane/BartLM/checkpoints/checkpoint_last.pt', 
                                 bpe="sentencepiece",
                                 sentencepiece_model="/home/gayane/BartLM/Bart/chemical/tokenizer/chem.model")
bart.eval()
bart.cuda(device=1)


input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")
smiles = list(load_indexed_dataset(
    f"{store_path}/{dataset}/processed/input0/{data_type}", input_dict))


test_label_path = f"{store_path}/{dataset}/processed/label/{data_type}"

# if task_type == 'classification':
    
#     target_dict = Dictionary.load(f"{store_path}/{dataset}/processed/label/dict.txt")
#     targets = list(load_indexed_dataset(test_label_path, target_dict))
# elif task_type == 'regression':
#     with open(f'{test_label_path}.label') as f:
#         lines = f.readlines()
#         targets = [float(x.strip()) for x in lines]
# if task_type == 'classification':
#     if len(dataset_js["class_index"])>1:
#         target_dict = list()
#         targets_list = list()
#         for i in range(len(dataset_js["class_index"])):
#             target_dict.append(Dictionary.load(f"{store_path}/{dataset}_{i}/processed/label/dict.txt"))
#             targets_list.append(list(load_indexed_dataset(test_label_path[i], target_dict[i])))
        
#     else: 
#         target_dict = Dictionary.load(f"{store_path}/{dataset}/processed/label/dict.txt")
#         targets = list(load_indexed_dataset(test_label_path, target_dict))
# elif task_type == 'regression':
#     with open(f'{test_label_path}.label') as f:
#         lines = f.readlines()
#         targets = [float(x.strip()) for x in lines]

# import re
# smi = []
# for sm in smiles:
#     smi.append(bart.decode(torch.tensor(sm)))

# _input = pd.read_csv(f"""/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/BBBP/raw/{data_type}.input""", names=['SMILES'], header=None)
# li = _input["SMILES"].to_list()
# # print(li[:2])
# l = _input["SMILES"].to_list()
# for sml in range(len(li)):
#     # li[sml] = li[sml][:122] + "<mask>"
#     if "(" in li[sml]:
#         li[sml] = li[sml][:122].replace('(', "<mask>", 1)



inputs = ['C'] #, '[H+].C2=C1C(OC(=NC1=CC=C2Cl)NCC)(C3=CC=CC=C3)C.[Cl-]', 'C(=O<mask>(OC(C)(C)C)CCCc1ccc(<mask>1)N(CCCl)CCCl', 'C<mask>C(Cl)C=CC3=C1C(C2=CC=CC=C2)SC(=N3)NCC', "C1=C(<mask>)C=CC3=C1C(C2=CC=CC=C2)SC(=N3)NCC", "C1=C<mask>Cl)C=CC3=C1C(C2=CC=CC=C2)SC(=N3)NCC"] # li #
results = bart.fill_mask(masked_inputs = inputs,  topk=3, match_source_len=True, beam=1)
assert len(inputs) == len(results), "different sizes?"


d = {"masked SMILES": inputs, "filled SMILES": results}
df = pd.DataFrame(d)
df.to_csv("/home/gayane/BartLM/fairseq/scripts/filled_mask_pretrained_match_source_len_False.csv")


