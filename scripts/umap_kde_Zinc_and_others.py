import matplotlib.pyplot as plt
import matplotlib.patches as  mpatches

from itertools import chain
from pathlib import Path
import seaborn as sns
import pandas as pd 
import numpy as np
import torch
import json
import umap
import os


pretrained_model = True
pretr = "-pretrained" if pretrained_model else ""
path = "/mnt/good/gayane/data/data_load_folder"

log_save_dir = Path(f"./umap-graph-all{pretr}/")
log_save_dir.mkdir(parents=True, exist_ok=True)
n_neighbors = [40]
min_dists = [0.4]
# tar = [0 if i[0].item() == 5 else 1 for i in targets]


# col_train = ["orange" if i==1 else "pink" for i in df[df['level_0']=="train"].target.to_list()]
# col_valid = ["darkred" if i==1 else "dimgrey" for i in df[df['level_0']=="valid"].target.to_list()]
# col_test = ["green" if i==1 else "darkblue" for i in df[df['level_0']=="test"].target.to_list()]

color = ["gray", "black", "blue", "red", "green", "greenyellow", "violet"]


leg = []
# col = list(chain.from_iterable([col_train, col_valid, col_test]))


# dataset_len = [475196, 2039, 1478, 1478, 1427, 641]
# in_ = [(7063, 482259), (0, 2039), (2039, 3517), (3517, 4995), (4995, 6422), (6422, 7063)]

# dataset_len = [2039, 1478, 7831, 1427, 641, 1128, 475196, 4200, 41127]
# in_ = [(14544, 489740), (0, 2039), (2039, 3517), (3517, 11348), (11348, 12775), (12775, 13416), (13416, 14544), (489740, 493940), (493940, 535067)]


# dataset_len = [475196, 2039, 1478, 7831, 1427, 641]
dataset_name_list = [ "BBBP", "clintox", "Tox21", "SIDER", "Genotoxicity", "esol", "ZINC", "lipo", "HIV"] 

in_ = [(14544, 489740), (0, 2039), (2039, 3517), (3517, 11348), (11348, 12775), (12775, 13416), (13416, 14544), (489740, 493940), (493940, 535067)]


dataset_name_list = ["BBBP", "clintox", "Tox21", "SIDER", "Genotoxicity", "esol", "ZINC"] 
in_ = [(0, 2039), (2039, 3517), (3517, 11348), (11348, 12775), (12775, 13416), (13416, 14544), (14544, 489740)]

np_filename = f'{path}/np_{dataset_name_list[0]}_pretrainedTrue.npy'
df_filename = f'{path}/df_{dataset_name_list[0]}_pretrainedTrue.csv'
print(dataset_name_list[0])
X_ = np.load(np_filename)
df_0 = pd.read_csv(df_filename) 
dir_name = '_'.join(dataset_name_list)
os.system(f"mkdir -p {path}/{dir_name}")
data_len = 489740
for n_neighbor in n_neighbors:
    for min_dist in min_dists:

        np_filename = f'{path}/{dir_name}/np_pretrainedTrue_{data_len}_umap_{n_neighbor}_{min_dist}.npy'
        df_filename = f'{path}/{dir_name}/df_pretrainedTrue_{data_len}_umap_{n_neighbor}_{min_dist}.csv'
        filename = ""
        if not os.path.exists(np_filename):
            for i, data_name in enumerate(dataset_name_list):
                df_data = pd.DataFrame()
                filename = filename + f"_{data_name}"
                if i == 0: 
                    X = X_
                    df_ = df_0
                else:
                    print("Create umap matrix part")

                    np_filename_ = f'{path}/np_{data_name}_pretrainedTrue.npy'
                    df_filename_ = f'{path}/df_{data_name}_pretrainedTrue.csv'
                    print(data_name)
                    X_data = np.load(np_filename_)
                    if os.path.exists(df_filename_):
                        df_data = pd.read_csv(df_filename_)

                    Zinc = data_name == "ZINC"
                    n = X_data.shape[0]
                    tp = ["train"]*n if Zinc else None
                    tr = np.zeros(n) if Zinc else None
                    X_data = X_data.reshape(n, 1024) if Zinc else X_data
                    df_data["target"] = tr if Zinc else df_data["target"]
                    df_data["level_0"] = tp if Zinc else df_data["level_0"]

                    print(X_data.shape, X.shape)
                    X = np.concatenate((X, X_data))
                    df_ = pd.concat([df_, df_data], ignore_index=True, sort=False)

                    if data_name == "ZINC":
                        df_data.to_csv(f'{path}/df_{data_name}_pretrainedTrue.csv')
            print(f"---------- neighbor:  {n_neighbor}, minimum distance: {min_dist} ----------")
            data_len = X.shape[0]
            reducer = umap.UMAP(
                n_components=2, min_dist=min_dist, n_neighbors=n_neighbor, transform_seed=5, verbose=False
            ).fit(X[:data_len])
            
            X_umap_ = reducer.transform(X)
            print(f"Saving to {np_filename}")
            data = {
                    "x": X_umap_[:, 0].tolist(),
                    "y": X_umap_[:, 1].tolist(),
                    "label": df_["target"].values,
                    "split": df_["level_0"].values,

                }
            print(f"Saving to {np_filename}")
            np.save(file=np_filename, arr=X_umap_)

            df_ = pd.DataFrame(data=data)
            print(f"Saving to {df_filename}")
            df_.to_csv(df_filename)
            
        else:

            print(f"Reading from {df_filename}")
            df = pd.read_csv(df_filename)
            print(f"Reading from {np_filename}")
            X = np.load(np_filename)
        
        fig, ax = plt.subplots(figsize=(8,8))
        # X_filter = X[(X[:,0] >= -2.3) & (X[:,0] <= 2.3) & (X[:,1] >= 1)]
        # df_filter = df[(X[:,0] >= -2.3) & (X[:,0] <= 2.3) & (X[:,1] >= 1)]
        Zinc_level = 70

        # dataset_name_list__ = ["ZINC", "HIV", "Tox21", "lipo", "BBBP", "clintox", "SIDER", "Genotoxicity", "esol"] 

        # in_ = [(14544, 489740), (493940, 535067), (3517, 11348), (489740, 493940), (0, 2039), (2039, 3517), (11348, 12775), (12775, 13416), (13416, 14544)]
        dataset_name_list__ = ["ZINC", "BBBP", "ClinTox", "Tox21", "SIDER", "Micronucleus Assay", "ESOL"] 

        in_ = [(14544, 489740), (0, 2039), (2039, 3517), (3517, 11348), (11348, 12775), (12775, 13416), (13416, 14544)]

        for i, data_name in enumerate(dataset_name_list__):
            if data_name == "ZINC":
                x = df["x"].values[in_[i][0] : in_[i][1]]
                y =  df["y"].values[in_[i][0] : in_[i][1]]

                # x_filter = x[(x >= 1) & (x <= 8) & (y >= 0) & (y <= 9)]
                # y_filter = y[(x >= 1) & (x <= 8) & (y >= 0) & (y <= 9)]
                # ax = sns.kdeplot(x_filter, y_filter, alpha=0.6, thresh=.2, levels=Zinc_level, label=f"{data_name}", color=color[i], cmap="viridis")
                ax = sns.kdeplot(x, y, alpha=0.6, thresh=.2, levels=Zinc_level, label=f"{data_name}", color=color[i])
            else:
                x = df["x"].values[in_[i][0] : in_[i][1]]
                y = df["y"].values[in_[i][0] : in_[i][1]]
                ax = sns.kdeplot(x, y, alpha=0.6, thresh=.2, levels=3, label=f"{data_name}", color=color[i])
            # ax = sns.kdeplot(df[df["label"] ==1]["x"].values, df[df["label"] ==1]["y"].values,  alpha=0.6, thresh=.2, levels=6, label=f"{data_name}_1")

        handles = [mpatches.Patch(facecolor=color[i], label=dataset_name) for i, dataset_name in enumerate(dataset_name_list__)]
        plt.legend(handles=handles)    # leg.append(f"{data_name}_1")
        plt.savefig(f"umap_{n_neighbor}_{min_dist}_Zinc_level_{Zinc_level}.png")
        print(f"Finish saving: umap_{n_neighbor}_{min_dist}_Zinc_level_{Zinc_level}.png")
        plt.savefig(f"umap_{n_neighbor}_{min_dist}_Zinc_level_{Zinc_level}.pdf")
        print(f"Finish saving: umap_{n_neighbor}_{min_dist}_Zinc_level_{Zinc_level}.pdf")
        

