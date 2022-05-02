from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, classification_report, mean_squared_error, confusion_matrix
from fairseq.data.data_utils import load_indexed_dataset
from sklearn.linear_model import ridge_regression
from fairseq.models.bart import BARTModel
from fairseq.data import Dictionary
import torch.nn.functional as F 
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch
import json
import os


def multi_task_predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False):
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    features = self.extract_features(tokens.to(device=self.device))
    sentence_representation = features[
        tokens.eq(self.task.source_dictionary.eos()), :
    ].view(features.size(0), -1, features.size(-1))[:, -1, :]
    logits = list()
    for i in range(len(dataset_js["class_index"])>1): 
        logits.append(self.model.classification_heads[head+str(i)](sentence_representation))
    if return_logits:
        return logits
    probabies = list()
    for i in range(len(dataset_js["class_index"])>1): 
        probabies.append(F.log_softmax(logits[i], dim=-1))
    return probabies


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', required=True,
                        help='dataset name.')
parser.add_argument('--subtask', required=True,
                        help='subtask')
parser.add_argument('--warmup-update', required=True,
                        help='warmup update')
parser.add_argument('--total-number-update', required=True,
                        help='total number update')
parser.add_argument('--lr', default=3e-5,
                        help='learning rate')
args = parser.parse_args()

dataset = args.dataset_name if args.dataset_name in set(["BBBP", "BACE", "HIV"]) else f"{args.dataset_name}_{args.subtask}"

store_path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data"
model = f"{store_path}/{dataset}/processed"

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

bart = BARTModel.from_pretrained(model,  checkpoint_file = f"/mnt/good/gayane/data/chkpt/{dataset}_bs_16_lr_{args.lr}_totalNum_{args.total_number_update}_warmup_{args.warmup_update}/checkpoint_best.pt", 
                                 bpe="sentencepiece",
                                 sentencepiece_model="/home/gayane/BartLM/Bart/chemical/tokenizer/chem.model")
bart.eval()
bart.cuda(device=1)


input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")
smiles = list(load_indexed_dataset(
    f"{store_path}/{dataset}/processed/input0/valid", input_dict))

if len(dataset_js["class_index"])>1:
    test_label_path = list()
    for i in range(len(dataset_js["class_index"])):
        test_label_path.append(f"{store_path}/{dataset}/processed/label{i}/valid")

else:
    test_label_path = f"{store_path}/{dataset}/processed/label/valid"

if task_type == 'classification':
    if len(dataset_js["class_index"])>1:
        target_dict = list()
        targets_list = list()
        for i in range(len(dataset_js["class_index"])):
            target_dict.append(Dictionary.load(f"{store_path}/{dataset}/processed/label{i}/dict.txt"))
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
if len(dataset_js["class_index"])>1:
    y_pred_list = list()
    y_list = list()
    for j in range(len(dataset_js["class_index"])):
        y_pred = list()
        y = list()
        for i, (smile, target) in tqdm(list(enumerate(zip(smiles, targets_list[j])))):
            smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2])))
            output = bart.predict(f'sentence_classification_head{j}', smile)
            target = target[0].item()
            y_pred.append(output[0][1].exp().item())
            y.append(target - 4)
        y_pred_list.append(y_pred)
        y_list.append(y)
else:
    for i, (smile, target) in tqdm(list(enumerate(zip(smiles, targets)))):
        smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2])))  
        if task_type =="classification":
            output = bart.predict('sentence_classification_head', smile)
            target = target[0].item()
            y_pred.append(output[0][1].exp().item())
            y.append(target - 4)
            
        elif task_type == "regression":
            output = bart.predict('sentence_classification_head', smile, return_logits=True)
            y_pred.append(output[0][0].item())
            y.append(target)

if task_type == 'classification':
    if len(dataset_js["class_index"]) >1:
        if dataset == "muv" or dataset == "pcba" :
            prc_auc_list = list()
            for i in range(len(dataset_js["class_index"])):
                precision, recall, thresholds = precision_recall_curve(y_list[i], y_pred_list[i])
                prc_auc_list.append(auc(recall, precision))
                print("PRC_AUC_SCORE: ", auc(recall, precision))
                y_pred_binary = np.array(y_pred_list[i]) > 0.5
                print("Confusion matrix:")
                print(confusion_matrix(y_list[i], y_pred_binary))
            print("PRC_AUC_SCORE_MEAN: ", np.mean(prc_auc_list))
        else:
            roc_auc_list = list()
            for i in range(len(dataset_js["class_index"])):
                roc_auc_list.append(roc_auc_score(y_list[i], y_pred_list[i]))
                print("ROC_AUC_SCORE: ", roc_auc_score(y_list[i], y_pred_list[i]))
                y_pred_binary = np.array(y_pred_list[i]) > 0.5
                print("Confusion matrix:")
                print(confusion_matrix(y_list[i], y_pred_binary))
            print("ROC_AUC_SCORE_MEAN: ", np.mean(roc_auc_list))
            print(roc_auc_list)

        

    else: 
        print("ROC_AUC_SCORE: ", roc_auc_score(y, y_pred))
        y_pred_binary = np.array(y_pred) > 0.5
        print(classification_report(y, y_pred_binary))
        print("Confusion matrix:")
        print(confusion_matrix(y, y_pred_binary))
else:
    y_prd = [(ma -mi)*x +mi  for x in y_pred]
    y_l = [(ma -mi)*x + mi  for x in y]
    df = pd.DataFrame(data={"y_l": y, "y_pred": y_prd, "y_l_scale": y_l, "y_pred_scale": y_pred})
    print(mean_squared_error([(ma -mi)*x + mi  for x in y], [(ma -mi)*x +mi  for x in y_pred]))
