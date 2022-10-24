from tkinter import N
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from scripts.process import getMurcoScaffoldList
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import chain
from pathlib import Path
from rdkit import Chem
from sklearn import metrics
import torch.nn as nn
from tqdm import tqdm
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
import umap.plot
import logging
import torch
import json
import umap
import os

# os.system['MKL_THREADING_LAYER'] = 'GNU'
# os.system('CUDA_LAUNCH_BLOCKING=1')
# os.environ["PYTORCH_CUDA_ALLOC_CONF"]="5"



def generateMurcoScaffold(df):
        include_chirality = False
        _scaff = []
        for i in range(len(df)):
                mol = Chem.MolFromSmiles(df['SMILES'][i])
                _scaff.append(MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality))
        df["MurckoScaffold"] = _scaff
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        return df
def choose_n_to_k(n):
    l = list()
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            l.append((i-1, j-1))
    return l

def add_edge_to_graph(G, e1, e2):
    G.add_edge(e1, e2)



dataset = 'esol'
pretrained_model = True
# dataset = dataset_name if dataset_name in set(["BBBP", "BACE", "HIV"]) else f"{dataset_name}_{args.subtask}"

df_filename = f"/mnt/good/gayane/data/data_load_folder/df_{dataset}_pretrained{pretrained_model}.csv"
np_filename = f"/mnt/good/gayane/data/data_load_folder/np_{dataset}_pretrained{pretrained_model}.npy"

def generate_df_np():
    from fairseq.data.data_utils import load_indexed_dataset
    from fairseq.models.bart import BARTModel
    from fairseq.data import Dictionary
    import torch.nn.functional as F 

    pretrained = "" if pretrained_model else "/input0"
    store_path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data"
    model = f"{store_path}/{dataset}/processed{pretrained}"

    with open('/home/gayane/BartLM/fairseq/scripts/datasets.json') as f:
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

    # warmup = 163
    # totNumUpdate = 1020
    # lr = '3e-5'
    # chkpt_path = '/mnt/good/gayane/data/chkpt/BBBP_bs_16_dropout_0.1_lr_3e-5_totalNum_1020_warmup_163_noise_type_uniform_r3f_lambda_1.0/checkpoint_best.pt'
    chkpt_path = '/home/gayane/BartLM/checkpoints/checkpoint_last.pt'
    # chkpt_path = f"/mnt/good/gayane/data/chkpt/{dataset}_bs_16_lr_{lr}_totalNum_{totNumUpdate}_warmup_{warmup}/checkpoint_last.pt"
    print(chkpt_path)  # BACE_bs_16_lr_3e-5_totalNum_1135_warmup_181/ in test 
    bart = BARTModel.from_pretrained(model,  checkpoint_file = chkpt_path, 
                                    bpe="sentencepiece",
                                    sentencepiece_model="/home/gayane/BartLM/Bart/chemical/tokenizer/chem.model")

    input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")

    bart.eval()
    bart.cuda(device=1)


    data_type = 'train'
    def get_data(data_type):
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
            smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2]))).to("cuda:1")  
            if task_type =="classification":
                # output = bart.predict('sentence_classification_head', smile)
                # target = target[0].item()
                # y_pred.append(output[0][1].exp().item())
                # y.append(target - 4)
                sm.append(smile)
                
            elif task_type == "regression":
                # output = bart.predict('sentence_classification_head', smile, return_logits=True)
                # y_pred.append(output[0][0].item())
                # y.append(target)
                sm.append(smile)
        return sm, targets #, y_pred


    sm_train, targets_train = get_data("train") # , y_pred_train
    sm_valid, targets_valid = get_data("valid") # , y_pred_valid
    sm_test, targets_test = get_data("test") # , y_pred_test


    # sm_train = get_data("train")

    # y_pred_train = pd.DataFrame({"y_pred":y_pred_train})
    # y_pred_valid = pd.DataFrame({"y_pred":y_pred_valid})
    # y_pred_test = pd.DataFrame({"y_pred":y_pred_test})

    smi = []

    targets_train = [i for i in targets_train]
    targets_valid = [i for i in targets_valid]
    targets_test = [i for i in targets_test]

    # targets_train = [0 if i[0].item() == 5 else 1 for i in targets_train]
    # targets_valid = [0 if i[0].item() == 5 else 1 for i in targets_valid]
    # targets_test = [0 if i[0].item() == 5 else 1 for i in targets_test]

    _target_train = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/{dataset}/raw/train.target", names=['target'], header=None)
    _target_valid = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/{dataset}/raw/valid.target", names=['target'], header=None)
    _target_test = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/{dataset}/raw/test.target", names=['target'], header=None)
    _input_train = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/{dataset}/raw/train.input", header=None)
    _input_valid = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/{dataset}/raw/valid.input", header=None)
    _input_test = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/{dataset}/raw/test.input", header=None)


    train_df = pd.concat([_input_train, _target_train], axis=1, join="inner") #, y_pred_train
    train_df = train_df.rename(columns={0: "SMILES"})
    valid_df = pd.concat([_input_valid, _target_valid], axis=1, join="inner") # , y_pred_valid
    valid_df = valid_df.rename(columns={0: "SMILES"})
    test_df = pd.concat([_input_test, _target_test], axis=1, join="inner") # , y_pred_test
    test_df = test_df.rename(columns={0: "SMILES"})


    # train_df = _input_train

    train_df = getMurcoScaffoldList(train_df, "SMILES", False)
    valid_df = getMurcoScaffoldList(valid_df, "SMILES", False)
    test_df = getMurcoScaffoldList(test_df, "SMILES", False)
    print("finished get scaffold")

    df =  pd.concat([train_df, valid_df, test_df],
                keys=['train', 'valid', 'test']).reset_index()

    # df = train_df



    len_dataset = len(df)


    print(f"Saving to {df_filename}")
    df.to_csv(df_filename)


    def get_feautures(sm):
        umap_X = []
        with torch.no_grad():
            for i in sm:
                last_layer_features = bart.extract_features(i)
                last_layer_feat_mean = torch.mean(last_layer_features, 1)
                umap_X.append(last_layer_feat_mean.to("cpu"))
        return umap_X

    print("starting extract train data")
    umap_X_train = get_feautures(sm_train)
    print("starting extract validation data")
    umap_X_valid = get_feautures(sm_valid)
    print("starting extract test data")
    umap_X_test = get_feautures(sm_test)


    umap_X_train = [i.numpy() for i in umap_X_train]
    umap_X_valid = [i.numpy() for i in umap_X_valid]
    umap_X_test = [i.numpy() for i in umap_X_test]


    X_list = [umap_X_train, umap_X_valid, umap_X_test]
    X = list(chain.from_iterable(X_list))
    X = np.array(X)
    X = X.reshape(len(X), 1024)

    print(f"Saving to {np_filename}")
    np.save(np_filename, X)

    return df, X


# if os.path.exists(np_filename):
#     generate_df_np()

if not os.path.exists(np_filename):
    generate_df_np()

print(f"Reading from {df_filename}")
df = pd.read_csv(df_filename)
print(f"Reading from {np_filename}")
X = np.load(np_filename)
