from tkinter import N
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import matplotlib.pyplot as plt
from itertools import chain
from pathlib import Path
from rdkit import Chem
from sklearn import metrics
from sklearn.manifold import TSNE
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



dataset_name = 'BACE'
len_dataset = 1513
pretrained_model = True
# dataset = dataset_name if dataset_name in set(["BBBP", "BACE", "HIV"]) else f"{dataset_name}_{args.subtask}"
dataset = dataset_name

df_filename = f"df_{dataset_name}_pretrained{pretrained_model}.csv"
np_filename = f"np_{dataset_name}_pretrained{pretrained_model}.npy"

def generate_df_np():
    from fairseq.data.data_utils import load_indexed_dataset
    from fairseq.models.bart import BARTModel
    from fairseq.data import Dictionary
    import torch.nn.functional as F 

    pretrained = "/input0" if pretrained_model else ""
    store_path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data"
    model = f"{store_path}/{dataset}/processed{pretrained}"

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

    # y_pred_train = pd.DataFrame({"y_pred":y_pred_train})
    # y_pred_valid = pd.DataFrame({"y_pred":y_pred_valid})
    # y_pred_test = pd.DataFrame({"y_pred":y_pred_test})

    smi = []


    def get_feautures(sm):
        umap_X = []
        # torch.cuda.empty_cache()
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


    targets_train = [0 if i[0].item() == 5 else 1 for i in targets_train]
    targets_valid = [0 if i[0].item() == 5 else 1 for i in targets_valid]
    targets_test = [0 if i[0].item() == 5 else 1 for i in targets_test]

    _target_train = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/BBBP/raw/train.target", names=['target'], header=None)
    _target_valid = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/BBBP/raw/valid.target", names=['target'], header=None)
    _target_test = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/BBBP/raw/test.target", names=['target'], header=None)
    _input_train = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/BBBP/raw/train.input", header=None)
    _input_valid = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/BBBP/raw/valid.input", header=None)
    _input_test = pd.read_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/BBBP/raw/test.input", header=None)


    train_df = pd.concat([_input_train, _target_train], axis=1, join="inner") #, y_pred_train
    train_df = train_df.rename(columns={0: "SMILES"})
    valid_df = pd.concat([_input_valid, _target_valid], axis=1, join="inner") # , y_pred_valid
    valid_df = valid_df.rename(columns={0: "SMILES"})
    test_df = pd.concat([_input_test, _target_test], axis=1, join="inner") # , y_pred_test
    test_df = test_df.rename(columns={0: "SMILES"})

    train_df = generateMurcoScaffold(train_df)
    valid_df = generateMurcoScaffold(valid_df)
    test_df = generateMurcoScaffold(test_df)
    print("finished get scaffold")

    df = pd.concat([train_df, valid_df, test_df],
                keys=['train', 'valid', 'test']).reset_index()

    X_list = [umap_X_train, umap_X_valid, umap_X_test]
    X = list(chain.from_iterable(X_list))
    X = np.array(X)
    X = X.reshape(len_dataset, 1024)

    print(f"Saving to {df_filename}")
    df.to_csv(df_filename)
    print(f"Saving to {np_filename}")
    np.save(np_filename, X)

    return df, X

if not os.path.exists(np_filename):
    generate_df_np()

print(f"Reading from {df_filename}")
df = pd.read_csv(df_filename)
print(f"Reading from {np_filename}")
X = np.load(np_filename)

exit()

subset = "train"

pretr = "-pretrained" if pretrained_model else ""


log_save_dir = Path(f"./umap-graph-all{pretr}/")
log_save_dir.mkdir(parents=True, exist_ok=True)
n_neighbors = [100] # [10, 40, 80, 100]
min_dists = [0.8] # [0.4, 0.6, 0.8, 1]
# tar = [0 if i[0].item() == 5 else 1 for i in targets]


col_train = ["orange" if i==1 else "pink" for i in df[df['level_0']=="train"].target.to_list()]
col_valid = ["darkred" if i==1 else "dimgrey" for i in df[df['level_0']=="valid"].target.to_list()]
col_test = ["green" if i==1 else "darkblue" for i in df[df['level_0']=="test"].target.to_list()]


col = list(chain.from_iterable([col_train, col_valid, col_test]))
node_shape = ["o" if i==1 else "v" for i in df.target.to_list()]

for n_neighbor in n_neighbors:
    for min_dist in min_dists:
        print(f"---------- neighbor:  {n_neighbor}, minimum distance: {min_dist} ----------")
        reducer = umap.UMAP(
            n_components=2, min_dist=min_dist, n_neighbors=n_neighbor, transform_seed=5, verbose=False
        ).fit(X)
        X_umap_ = reducer.transform(X)
        data = {
                "x": X_umap_[:, 0].tolist(),
                "y": X_umap_[:, 1].tolist(),
                "label": df["target"].values,
                # "y_pred": df["y_pred"].values,
                "split": df["level_0"].values,
                "MurckoScaffold": df["MurckoScaffold"].values,
                "col": col
            }
        df_ = pd.DataFrame(data=data)
        
        nodes = np.arange(df_.shape[0])
        
        G = nx.Graph()
        G.add_nodes_from(nodes)
        pos = [[df_['x'][i], df_['y'][i]] for i in range(df_.shape[0])]

        all_datasets = list(set(df_["MurckoScaffold"]))
        edges = []
        distance = np.zeros((len(df), len(df))) + 100000

        stats = {
            "size": [],
            "split": [],
            "acc": [],
        }

        for s in all_datasets:
            A = df_[df_['MurckoScaffold'] == s]

            n_ = A.shape[0] # number of scaffold types
            if n_ == 0:
                continue

            acc = 100 * metrics.accuracy_score(A['label'], A["y_pred"] <= 0.5)
            stats['size'].append(n_)
            stats['split'].append(A['split'].values[0])
            stats['acc'].append(acc)

            # if n_ <= 15:
            #     continue
            # idx = [k for k in range(n_)]
            # for l in [0,1]:
                # A_same_label = A[A["label"] == l]
                # n_ = A_same_label.shape[0] # number of scaffold types
            # idx = [k for k in range(n_)]
            # A['idx'] = idx
            ed_p = A.index.to_list()
            for i in range(len(ed_p)):
                for j in range(i + 1, len(ed_p)-1):
                    edges.append((ed_p[i], ed_p[j]))
                    distance[ed_p[i], ed_p[j]] = 1
        
        df_stats = pd.DataFrame(stats)
        df_stats.to_csv(f"{dataset_name}_stats.csv")

        # print(edges)
        for i in range(len(edges)):
        
            add_edge_to_graph(G, edges[i][0], edges[i][1])
        
        print("Starting the layout algorithm")
        pos = nx.spring_layout(G, iterations=50, k=0.15)
        # tsne = TSNE(2, metric="precomputed").fit_transform(distance)
        # pos = tsne
        print("...Done")


        fig, ax = plt.subplots()

        nx.draw_networkx(G, pos=pos, ax=ax, node_size=10, edge_color="black", 
        linewidths=0.5, node_color=col, alpha=0.4, width=0.2,  node_shape='o', with_labels=False)
        
        plt.axis("on")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False)
        plt.xlabel("Orange and pink are from train set, red and gray are from validation " + "\n" + "and green and darkblue are from test. " + "\n" + "Brown, orange and green have 1 label and pink, darkred and darkblue have 0 label")
        plt.savefig(log_save_dir.joinpath(f"{subset}-{n_neighbor}-{min_dist}.pdf"))
        plt.savefig(log_save_dir.joinpath(f"{subset}-{n_neighbor}-{min_dist}.png"))

        plt.clf()

        # plt.show()