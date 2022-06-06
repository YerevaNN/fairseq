import os
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

PRETRAINED_MODEL_NAME = "checkpoints.checkpoint_last"


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
base_dir = Path("/home/tmyn/toxo/un-logs")
for pool in pools:
    pool_dir = base_dir.joinpath(pool)
    task_names = [name for name in os.listdir(pool_dir) if os.path.isdir(pool_dir.joinpath(name))]

    records_pretrained = {}
    records_finetuned = {}
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
                records_pretrained[task_name] = {
                    "x": num_features,
                    "y": auc_roc,
                }
            else:
                records_finetuned[task_name] = {
                    "x": num_features,
                    "y": auc_roc,
                }

    # pretrained
    for d_name, data in records_pretrained.items():
        sns.lineplot(x=data["x"], y=data["y"], marker='o', linewidth=2.0, label=d_name)

    plt.legend(loc="lower right")
    plt.xlabel("Number of Features")
    plt.xscale('log')
    plt.ylabel("AUC-ROC score")
    plt.savefig(f"nf-vs-auc/{pool}.pretrained.png")
    plt.cla()

    # finetuned
    for d_name, data in records_finetuned.items():
        sns.lineplot(x=data["x"], y=data["y"], marker='o', linewidth=2.0, label=d_name)

    plt.legend(loc="lower right")
    plt.xlabel("Number of Features")
    plt.xscale('log')
    plt.ylabel("AUC-ROC score")
    plt.savefig(f"nf-vs-auc/{pool}.finetuned.png")
    plt.cla()

    # combined
    for d_name, data in records_pretrained.items():
        sns.lineplot(x=data["x"], y=data["y"], marker='o', linewidth=2.0, label=d_name)

    for d_name, data in records_finetuned.items():
        sns.lineplot(x=data["x"], y=data["y"], marker='o', linewidth=2.0, label=d_name, linestyle='--')

    plt.legend(loc="lower right")
    plt.xlabel("Number of Features")
    plt.xscale('log')
    plt.ylabel("AUC-ROC score")
    plt.savefig(f"nf-vs-auc/{pool}.combined.png")
    plt.cla()
