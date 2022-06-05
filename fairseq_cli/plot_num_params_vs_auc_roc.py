import os
import matplotlib.pyplot as plt
from pathlib import Path


def get_color(task_name):
    if task_name == "clintox_0":
        return "#5dba7c"
    elif task_name == "clintox_1":
        return "#0d4d23"
    elif task_name == "Genotoxicity":
        return "red"
    elif task_name == "SIDER_0":
        return "blue"
    elif task_name == "BBBP":
        return "purple"
    else:
        return "black"


PRETRAINED_MODEL_NAME = "checkpoints.checkpoint_last"

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
                    "color": get_color(task_name)
                }
            else:
                records_finetuned[task_name] = {
                    "x": num_features,
                    "y": auc_roc,
                    "color": get_color(task_name)
                }
            
    fig, ax = plt.subplots()
    for d_name, data in records_pretrained.items():
        plt.plot(data["x"], data["y"], marker='.', color=data["color"], label=d_name)
    plt.legend(loc="upper right")
    plt.title(f"Pre-train: {pool}")
    plt.xlabel("Number of Features")
    plt.ylabel("AUC-ROC score")
    plt.savefig(f"nf-vs-auc/{pool}.pretrained.png")
    plt.cla()

    for d_name, data in records_finetuned.items():
        plt.plot(data["x"], data["y"], marker='.', color=data["color"], label=d_name)
    plt.legend(loc="upper right")
    plt.title(f"Fine-tune: {pool}")
    plt.xlabel("Number of Features")
    plt.ylabel("AUC-ROC score")
    plt.savefig(f"nf-vs-auc/{pool}.finetuned.png")




# # pre-trained
# records_pretrained = {
#     "clintox_0": {
#         "x": [1, 4, 7, 11, 25, 42, 60, 91, 112],
#         "y": [41.52, 91.57, 98.71, 99.21, 99.40, 99.40, 99.40, 99.60, 99.50],
#         "color": "#5dba7c"
#     },
#     "clintox_1": {
#         "x": [1, 4, 8, 16, 26, 52, 67, 92, 116],
#         "y": [37.29, 92.80, 98.49, 98.49, 98.76, 98.76, 98.76, 99.02, 98.93],
#         "color": "#0d4d23"
#     },
#     "genotoxicity": {
#         "x": [1, 2, 9, 18, 28, 48, 72, 94, 144],
#         "y": [48.48, 73.36, 85.10, 85.61, 88.64, 91.29, 92.93, 93.94, 92.68],
#         "color": "red"
#     },
#     "sider_0": {
#         "x": [1, 4, 6, 15, 32, 66, 135, 216, 312],
#         "y": [51.64, 56.48, 60.00, 61.21, 62.70, 65.53, 66.52, 65.38, 67.36],
#         "color": "blue"
#     },
#     "bbbp": {
#         "x": [5, 6, 13, 28, 60, 93, 120, 213, 324],
#         "y": [86.88, 87.09, 89.01, 90.56, 91.63, 93.31, 93.58, 93.99, 93.78],
#         "color": "purple"
#     }
# }

# # finetuned
# records_fintuned = {
#     "clintox_0": {
#         "x": [1, 1, 3, 3, 8, 9, 14, 18, 36],
#         "y": [100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
#         "color": "#5dba7c"
#     },
#     "clintox_1": {
#         "x": [1, 2, 3, 3, 8, 13, 37, 39, 87],
#         "y": [99.29, 99.29, 99.20, 99.11, 99.11, 99.11, 99.20, 99.02, 98.93],
#         "color": "#0d4d23"
#     },
#     "genotoxicity": {
#         "x": [3, 6, 9, 15, 33, 51, 72, 101, 129],
#         "y": [92.55, 94.19, 95.20, 94.82, 93.31, 91.67, 90.91, 89.02, 87.37],
#         "color": "red"
#     },
#     "sider_0": {
#         "x": [1, 3, 10, 15, 36, 73, 123, 215, 321],
#         "y": [50.15, 59.78, 59.88, 60.73, 65.41, 67.95, 68.35, 67.50, 65.15],
#         "color": "blue"
#     },
#     "bbbp": {
#         "x": [3, 6, 11, 18, 24, 30, 39, 93, 150],
#         "y": [97.58, 97.60, 97.65, 97.72, 97.63, 97.60, 97.59, 97.62, 97.45],
#         "color": "purple"
#     }
# }
