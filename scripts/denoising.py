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
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
# os.system('CUDA_LAUNCH_BLOCKING=0')

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

warmup = 163
totNumUpdate = 1020
lr = '3e-5'
chkpt_path = '/mnt/good/gayane/data/chkpt/BBBP_bs_16_dropout_0.1_lr_3e-5_totalNum_1020_warmup_163_noise_type_uniform_r3f_lambda_1.0/checkpoint_best.pt'
# chkpt_path = f"/mnt/good/gayane/data/chkpt/{dataset}_bs_16_lr_{lr}_totalNum_{totNumUpdate}_warmup_{warmup}/checkpoint_last.pt"
print(chkpt_path)  # BACE_bs_16_lr_3e-5_totalNum_1135_warmup_181/ in test 
bart = BARTModel.from_pretrained(model,  checkpoint_file = "/home/gayane/BartLM/checkpoints/checkpoint_last.pt", 
                                 bpe="sentencepiece",
                                 sentencepiece_model="/home/gayane/BartLM/Bart/chemical/tokenizer/chem.model")
bart.eval()
bart.cuda(device=0)


data_type = 'test'

input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")
smiles = list(load_indexed_dataset(
    f"{store_path}/{dataset}/processed/input0/{data_type}", input_dict))


test_label_path = f"{store_path}/{dataset}/processed/label/{data_type}"

if task_type == 'classification':
    
    target_dict = Dictionary.load(f"{store_path}/{dataset}/processed/label/dict.txt")
    targets = list(load_indexed_dataset(test_label_path, target_dict))
elif task_type == 'regression':
    with open(f'{test_label_path}.label') as f:
        lines = f.readlines()
        targets = [float(x.strip()) for x in lines]
if task_type == 'classification':
    if len(dataset_js["class_index"])>1:
        target_dict = list()
        targets_list = list()
        for i in range(len(dataset_js["class_index"])):
            target_dict.append(Dictionary.load(f"{store_path}/{dataset}_{i}/processed/label/dict.txt"))
            targets_list.append(list(load_indexed_dataset(test_label_path[i], target_dict[i])))
        
    else: 
        target_dict = Dictionary.load(f"{store_path}/{dataset}/processed/label/dict.txt")
        targets = list(load_indexed_dataset(test_label_path, target_dict))
elif task_type == 'regression':
    with open(f'{test_label_path}.label') as f:
        lines = f.readlines()
        targets = [float(x.strip()) for x in lines]

y_pred = []
y = []
sm = []
for i, (smile, target) in tqdm(list(enumerate(zip(smiles, targets)))):
    smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2])))  
    if task_type =="classification":
        # output = bart.predict('sentence_classification_head', smile)
        # target = target[0].item()
        # y_pred.append(output[0][1].exp().item())
        # y.append(target - 4)
        sm.append(bart.decode(smile))
        
    elif task_type == "regression":
        # output = bart.predict('sentence_classification_head', smile, return_logits=True)
        # y_pred.append(output[0][0].item())
        # y.append(target)
        sm.append(bart.decode(smile))

smi = []



_input = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/BBBP/raw/{data_type}.input", names=['SMILES'], header=None)
print(len(_input))

li = _input["SMILES"].to_list()
l = _input["SMILES"].to_list()


# smi = []
# for sml in range(len(li)):
#     if "C" in li[sml]:
#         li[sml] = li[sml].replace('C', "", 1)
# for s in li:
#     smi.append(bart.decode(bart.encode(s)))

false_smile = []
false_recons = []
original = []
# for i in range(len(li)):
#     if smi[i] == li[i] and  l[i] == li[i]:
#         continue
#     false_smile.append(li[i])
#     false_recons.append(smi[i])
#     original.append(l[i])

original =    ["[H+].C2=C1C(OC(=NC1=CC=C2Cl)NCC)(C3=CC=CC=C3)C.[Cl-]", "[H+].C2=C1C(OC(=NC1=CC=C2Cl)NCC)(C3=CC=CC=C3)C.[Cl-]", "C1=C(Cl)C=CC3=C1C(C2=CC=CC=C2)SC(=N3)NCC", "C1=C(Cl)C=CC3=C1C(C2=CC=CC=C2)SC(=N3)NCC",]
false_smile = ["[H+].CC1C(OC(=NC1=CC=C2Cl)NCC)(C3=CC=CC=C3)C.[Cl-]",   "[H+].C2=C1COC(=NC1=CC=C2Cl)NCC)(C3=CC=CC=C3)C.[Cl-]",  "C1=C()C=CC3=C1C(C2=CC=CC=C2)SC(=N3)NCC",   "C1=CCl)C=CC3=C1C(C2=CC=CC=C2)SC(=N3)NCC", ]
for s in false_smile:
    smi.append(bart.decode(bart.encode(s)))
for s in original:
    false_recons.append(bart.decode(bart.encode(s)))

d = {"Original SMILES": original, "Reconstracted original SMILES": false_recons, "removed token": false_smile, "Reconstracted SMILES": smi}
df_bbbp = pd.DataFrame.from_dict(d)
# assert len(li) == len(smi), "different sizes?"


df_bbbp.to_csv("/home/gayane/BartLM/fairseq/scripts/encode_decode_remove_token.csv")


