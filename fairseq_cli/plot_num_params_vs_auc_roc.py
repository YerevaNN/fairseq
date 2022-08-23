import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
# PRETRAINED_MODEL_NAME = "checkpoints.checkpoint_last"
PRETRAINED_MODEL_NAME = "base.checkpoint_last"


rc_dict = {
    'figure.figsize': (12.8, 9.675),
    'figure.dpi': 300,
    'figure.titlesize': 35,
    'axes.titlesize': 35,
    'axes.labelsize': 35,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.grid': False,
    'axes.axisbelow': 'line',
    'axes.labelcolor': 'black',
    'figure.facecolor': 'white',
    'grid.color': '#ffffff',
    'grid.linestyle': '-',
    'text.color': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    #  'lines.solid_capstyle': <CapStyle.projecting: 'projecting'>,
    'patch.edgecolor': 'black',
    'patch.force_edgecolor': False,
    'image.cmap': 'viridis',
    'font.family': ['sans-serif'],
    'font.sans-serif': ['DejaVu Sans',
                        'Bitstream Vera Sans',
                        'Computer Modern Sans Serif',
                        'Lucida Grande',
                        'Verdana',
                        'Geneva',
                        'Lucid',
                        'Arial',
                        'Helvetica',
                        'Avant Garde',
                        'sans-serif'],
    'xtick.bottom': True,
    'xtick.top': False,
    'ytick.left': True,
    'ytick.right': True,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.right': True,
    'axes.spines.top': True}

sns.set(rc=rc_dict)

sns.set_palette(sns.color_palette("colorblind"))
sns.despine()


pools = ["avg", "last"]
base_dir = Path("un-logs")
output_dir = Path("nf-vs-auc")
output_dir.mkdir(parents=True, exist_ok=True)
for pool in pools:
    pool_dir = base_dir.joinpath(pool)
    task_names = [name for name in os.listdir(pool_dir) if os.path.isdir(pool_dir.joinpath(name))]

    records_pretrained = {"x": [], "y": [], "task": []}
    records_finetuned = {"x": [], "y": [], "task": []}
    for task_name in task_names:
        task_name_dir = pool_dir.joinpath(task_name)
        model_names = os.listdir(task_name_dir)
        for model_name in model_names:
            model_name_dir = task_name_dir.joinpath(model_name)

            Cs = [file_name for file_name in os.listdir(model_name_dir) if file_name.endswith('.txt')]
            Cs.sort()

            num_features = []
            auc_roc = []

            for C in Cs:
                with open(model_name_dir.joinpath(C), "r") as f:
                    lines = f.readlines()
                    num_features.append(int(lines[1].split(" ")[0]))
                    auc_roc.append(float(lines[4].split(" ")[0][:5]))

            if model_name_dir.stem == PRETRAINED_MODEL_NAME:
                records_pretrained["x"].append(num_features)
                records_pretrained["y"].append(auc_roc)
                records_pretrained["task"].append(task_name)
            else:
                records_finetuned["x"].append(num_features)
                records_finetuned["y"].append(auc_roc)
                records_finetuned["task"].append(task_name)

    # pretrained
    records_pretrained = pd.DataFrame.from_dict(records_pretrained)
    for task_name in records_pretrained.task:
        x = list(records_pretrained.loc[records_pretrained["task"] == task_name]["x"])[0]
        y = list(records_pretrained.loc[records_pretrained["task"] == task_name]["y"])[0]
        sns.lineplot(x=x, y=y, marker='o', linewidth=2.0, label=task_name)

    plt.legend(loc="lower right", fontsize=22)
    plt.xlabel("Number of Features")
    plt.xscale('log')
    plt.ylabel("AUC-ROC score")
    plt.tight_layout()
    plt.savefig(output_dir.joinpath(f"{pool}.pretrained.png"))
    plt.cla()

    # finetuned
    records_finetuned_dots = {"x": [1024, 1024, 1024, 1024, 1024],
                              "y": [99.2, 98.6, 96.3, 69.7, 99.0],
                              "task": task_names}
    records_finetuned_dots = pd.DataFrame.from_dict(records_finetuned_dots)
    sns.scatterplot(data=records_finetuned_dots, x="x", y="y", marker="H", s=120, hue="task", legend=None)

    records_finetuned = pd.DataFrame.from_dict(records_finetuned)
    for task_name in records_finetuned.task:
        x = list(records_finetuned.loc[records_finetuned["task"] == task_name]["x"])[0]
        y = list(records_finetuned.loc[records_finetuned["task"] == task_name]["y"])[0]
        sns.lineplot(x=x, y=y, marker='o', linewidth=2.0, label=task_name)

    plt.legend(loc="lower right", fontsize=22)
    plt.xlabel("Number of Features")
    plt.xscale('log')
    plt.ylabel("AUC-ROC score")
    plt.tight_layout()
    plt.savefig(output_dir.joinpath(f"{pool}.finetuned.png"))
    plt.cla()

    # # combined
    # for d_name, data in records_pretrained.items():
    #     sns.lineplot(x=data["x"], y=data["y"], marker='o', linewidth=2.0, label=d_name)

    # for d_name, data in records_finetuned.items():
    #     sns.lineplot(x=data["x"], y=data["y"], marker='o', linewidth=2.0, label=d_name, linestyle='--')

    # plt.legend(loc="lower right")
    # plt.xlabel("Number of Features")
    # plt.xscale('log')
    # plt.ylabel("AUC-ROC score")
    # plt.tight_layout()
    # plt.savefig(output_dir.joinpath(f"{pool}.combined.png"))
    # plt.cla()
