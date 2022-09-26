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
from tqdm import tqdm


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

color = ['lightgray', '#0000ff', '#ff1493', '#1e90ff', '#ffa500', '#00ff00', '#66cdaa']


leg = []
# col = list(chain.from_iterable([col_train, col_valid, col_test]))


# dataset_len = [475196, 2039, 1478, 1478, 1427, 641]
# in_ = [(7063, 482259), (0, 2039), (2039, 3517), (3517, 4995), (4995, 6422), (6422, 7063)]

# dataset_len = [2039, 1478, 7831, 1427, 641, 1128, 475196, 4200, 41127]
# in_ = [(14544, 489740), (0, 2039), (2039, 3517), (3517, 11348), (11348, 12775), (12775, 13416), (13416, 14544), (489740, 493940), (493940, 535067)]


# dataset_len = [475196, 2039, 1478, 7831, 1427, 641]
# dataset_name_list = [ "BBBP", "clintox", "Tox21", "SIDER", "Genotoxicity", "esol", "ZINC", "lipo", "HIV"] 

# in_ = [(14544, 489740), (0, 2039), (2039, 3517), (3517, 11348), (11348, 12775), (12775, 13416), (13416, 14544), (489740, 493940), (493940, 535067)]


# dataset_name_list = ["BBBP", "clintox", "Tox21", "SIDER", "Genotoxicity", "esol"]#, "ZINC965k"] 
# in_ = [(0, 2039), (2039, 3517), (3517, 11348), (11348, 12775), (12775, 13416), (13416, 14544)]#, (14544, 14544+965625)]


dataset_name_list = ["BBBP","clintox", "Tox21", "SIDER", "ZINC864k", "USPTO-50k"] 
in_ = [(0, 2039), (2039, 3517), (3517, 11348), (11348, 12775), (12775, 12775+864085), (12775+864085, 12775+864085+135605)]

# dataset_name_list= [ "USPTO-50k", "ZINC"]
# in_ = [(0, 135605), (135605, 610801)]

np_filename = f'{path}/np_{dataset_name_list[0]}_pretrainedTrue.npy'
df_filename = f'{path}/df_{dataset_name_list[0]}_pretrainedTrue.csv'
print(dataset_name_list[0])
X_ = np.load(np_filename)
# df_0 = pd.read_csv(df_filename) 
dir_name = '_'.join(dataset_name_list)
os.system(f"mkdir -p {path}/{dir_name}")
data_len = 610801
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
#                     df_ = df_0
                else:
                    print("Create umap matrix part")

                    np_filename_ = f'{path}/np_{data_name}_pretrainedTrue.npy'
                    df_filename_ = f'{path}/df_{data_name}_pretrainedTrue.csv'
                    print(data_name)
                    X_data = np.load(np_filename_)
                    if os.path.exists(df_filename_):
                        df_data = pd.read_csv(df_filename_)

                    Zinc = True ## ??? # data_name.startswith("ZINC")
                    n = X_data.shape[0]
                    tp = ["train"]*n if Zinc else None
                    tr = np.zeros(n) if Zinc else None
                    X_data = X_data.reshape(n, 1024) if Zinc else X_data
                    df_data["target"] = tr if Zinc else df_data["target"]
                    df_data["level_0"] = tp if Zinc else df_data["level_0"]

                    print(X_data.shape, X.shape)
                    X = np.concatenate((X, X_data))
#                     df_ = pd.concat([df_, df_data], ignore_index=True, sort=False)

#                     if Zinc:
#                         df_data.to_csv(f'{path}/df_{data_name}_pretrainedTrue.csv')
            print(f"---------- neighbor:  {n_neighbor}, minimum distance: {min_dist} ----------")
            data_len = X.shape[0]
            reducer = umap.UMAP(
                n_components=2, min_dist=min_dist, n_neighbors=n_neighbor, transform_seed=5, verbose=True
            ).fit(X[:data_len])
            
            X_umap_ = reducer.transform(X)
            print(f"Saving to {np_filename}")
            data = {
                    "x": X_umap_[:, 0].tolist(),
                    "y": X_umap_[:, 1].tolist(),
#                     "label": df_["target"].values,
#                     "split": df_["level_0"].values,

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
            print("X.shape", X.shape)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        # X_filter = X[(X[:,0] >= -2.3) & (X[:,0] <= 2.3) & (X[:,1] >= 1)]
        # df_filter = df[(X[:,0] >= -2.3) & (X[:,0] <= 2.3) & (X[:,1] >= 1)]

        # dataset_name_list__ = ["ZINC", "HIV", "Tox21", "lipo", "BBBP", "clintox", "SIDER", "Genotoxicity", "esol"] 
        # in_ = [(14544, 489740), (493940, 535067), (3517, 11348), (489740, 493940), (0, 2039), (2039, 3517), (11348, 12775), (12775, 13416), (13416, 14544)]
        
        
#         dataset_name_list__ = ["BBBP", "ClinTox", "Tox21", "SIDER", "Micronucleus Assay", "ESOL"] 
#         in_ = [ (0, 2039), (2039, 3517), (3517, 11348), (11348, 12775), (12775, 13416), (13416, 14544)]
        
#         dataset_name_list__ = ["ZINC965k", "BBBP", "ClinTox", "Tox21", "SIDER", "Micronucleus Assay", "ESOL"] 
#         in_ = [(14544, 14544+965625), (0, 2039), (2039, 3517), (3517, 11348), (11348, 12775), (12775, 13416), (13416, 14544)]

        dataset_name_list__ = ["ZINC864k", "USPTO-50k", "BBBP", "ClinTox", "Tox21", "SIDER"] 
        in_ = [(12775, 12775+864085), (12775+864085, 12775+864085+135605), (0, 2039), (2039, 3517), (3517, 11348), (11348, 12775)]

        # dataset_name_list__ = [ "ZINC", "USPTO-50k"]
        # in_ = [(135605, 610801), (0, 135605)]

#         dataset_name_list__ = [ "ZINC", "USPTO-50k: first input", "USPTO-50k: outher inputs", "USPTO-50k: output"]
#         in_ = [(135605, 610801), (0, 50016), (50016, 85589), (85589, 135605)]
        print(f"Plotting {len(dataset_name_list__)} datasets")
    
        limit = 10000
    
        for i, data_name in enumerate(tqdm(dataset_name_list__)):
            levels, thr, fill, alpha = 1, 0.05, False, 0.8
            if data_name.startswith("ZINC"):
                levels = 10
                thr = 0.001
                fill = True
                alpha = 0.7
            elif 'USPTO' in data_name:
                levels = 1
                fill = False
                alpha = 0.8
                
            x = df["x"].values[in_[i][0] : in_[i][1]][:limit]
            y =  df["y"].values[in_[i][0] : in_[i][1]][:limit]
            
#             thr = min(0.1, 1000.0 / (in_[i][1] - in_[i][0]))
            
            # x_filter = x[(x >= 1) & (x <= 8) & (y >= 0) & (y <= 9)]
            # y_filter = y[(x >= 1) & (x <= 8) & (y >= 0) & (y <= 9)]
            # ax = sns.kdeplot(x_filter, y_filter, alpha=0.6, thresh=.2, levels=Zinc_level, label=f"{data_name}", color=color[i], cmap="viridis")
            print(data_name, levels, thr)
            ax = sns.kdeplot(x[:], y[:], alpha=alpha, thresh=thr, fill=fill, levels=levels, label=f"{data_name}", color=color[i])
        
        handles = [mpatches.Patch(facecolor=color[i], label=dataset_name) for i, dataset_name in enumerate(dataset_name_list__)]
        plt.legend(handles=handles)    # leg.append(f"{data_name}_1")
        name = f"umap_{n_neighbor}_{min_dist}_{'_'.join(dataset_name_list__)}"
        plt.savefig(f"{name}.png")
        print(f"Finish saving: {name}.png")
        plt.savefig(f"{name}.pdf")
        print(f"Finish saving: {name}.pdf")
        

