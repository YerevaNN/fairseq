from fairseq.models.bart import BARTModel
from fairseq.data.data_utils import load_indexed_dataset
from fairseq.data import Dictionary
from sklearn.metrics import roc_auc_score, classification_report, mean_squared_error, confusion_matrix
from tqdm import tqdm
import os
import json
import torch
import numpy as np

# bbbp

dataset = "bbbp"
model = f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed"

with open('/home/gayane/BartLM/fairseq/scripts/datasets.json') as f:
    datasets_json = json.load(f)

dataset_js = datasets_json[dataset]
task_type = dataset_js['type']

if task_type == "regression":
    mi = dataset_js['minimum']
    ma = dataset_js['maximum']

os.system(f"mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/")
os.system(f"mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/")
os.system(f"mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/input0/")
os.system(f"mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/label/")

# os.system(f"cp /home/gayane/BartLM/{dataset}/processed/input0/dict.txt /home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/input0/")
# os.system(f"cp /home/gayane/BartLM/{dataset}/processed/label/dict.txt /home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}processed/label/")

bart = BARTModel.from_pretrained(model, checkpoint_file ='/mnt/bolbol/gayane/data/chkpt/bbbp3e-05/checkpoint_best.pt', 
                                 bpe="sentencepiece",
                                #  data_name_or_path= "/home/gayane/BartLM/Bart/chemical/evaluation_data/sampl/processed",
                                 sentencepiece_model="/home/gayane/BartLM/Bart/chemical/tokenizer/chem.model")
bart.eval()
bart.cuda(device=1)


input_dict = Dictionary.load(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/input0/dict.txt")

smiles = list(load_indexed_dataset(
    f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/input0/train", input_dict))
test_label_path = f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/label/train"

if task_type == 'classification':
    target_dict = Dictionary.load(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/label/dict.txt")
    targets = list(load_indexed_dataset(test_label_path, target_dict))
elif task_type == 'regression':
    with open(f'{test_label_path}.label') as f:
        lines = f.readlines()
        targets = [float(x.strip()) for x in lines]

y_pred = []
y = []
for i, (smile, target) in tqdm(list(enumerate(zip(smiles, targets)))):
    smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2])))
    output = bart.predict('sentence_classification_head', smile)
    
    if task_type =="classification":
        target = target[0].item()
        y_pred.append(output[0][1].exp().item())
        y.append(target - 4)
        
    elif task_type == "regression":
        y_pred.append(output[0][0].item())
        y.append(target)

if task_type == 'classification':
    print("ROC_AUC_SCORE: ", roc_auc_score(y, y_pred))
    y_pred_binary = np.array(y_pred) > 0.5
    print(classification_report(y, y_pred_binary))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred_binary))
else:
    print(mean_squared_error([(ma -mi)*x + mi  for x in y] ,  [(ma -mi)*x +mi  for x in y_pred]))


