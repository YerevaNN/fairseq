from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, classification_report, mean_squared_error, confusion_matrix
from fairseq.data.data_utils import load_indexed_dataset
from fairseq.models.bart import BARTModel
from fairseq.data import Dictionary
import torch.nn.functional as F 
from tqdm import tqdm
import numpy as np
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



dataset = "Ames"
model = f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed"

with open('/home/gayane/BartLM/fairseq/scripts/datasets.json') as f:
    datasets_json = json.load(f)

dataset_js = datasets_json[dataset]
task_type = dataset_js['type']

if task_type == "regression":
    mi = dataset_js['minimum']
    ma = dataset_js['maximum']

os.system(f"mkdir -p /home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/")
os.system(f"mkdir -p /home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/")
os.system(f"mkdir -p /home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/input0/")
os.system(f"mkdir -p /home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/label/")

bart = BARTModel.from_pretrained(model, checkpoint_file = f'/mnt/bolbol/gayane/data/chkpt/{dataset}8e-9/checkpoint_best.pt', 
                                 bpe="sentencepiece",
                                #  data_name_or_path= "/home/gayane/BartLM/Bart/chemical/evaluation_data/sampl/processed",
                                 sentencepiece_model="/home/gayane/BartLM/Bart/chemical/tokenizer/chem.model")
bart.eval()
bart.cuda(device=1)


input_dict = Dictionary.load(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/input0/dict.txt")

smiles = list(load_indexed_dataset(
    f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/input0/test", input_dict))

if len(dataset_js["class_index"])>1:
    test_label_path = list()
    for i in range(len(dataset_js["class_index"])):
        test_label_path.append(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/label{i}/test")

else:
    test_label_path = f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/label/test"

if task_type == 'classification':
    if len(dataset_js["class_index"])>1:
        target_dict = list()
        targets_list = list()
        for i in range(len(dataset_js["class_index"])):
            target_dict.append(Dictionary.load(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/label{i}/dict.txt"))
            targets_list.append(list(load_indexed_dataset(test_label_path[i], target_dict[i])))
        
        # print(type(targets_list))
        # targets_list = np.array(targets_list)
        # targets_list = np.transpose(targets_list)
    else: 
        target_dict = Dictionary.load(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{dataset}/processed/label/dict.txt")
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
   
   
    # trgt_list = list()

    # for i, (smile, target) in tqdm(list(enumerate(zip(smiles, targets_list)))):
    #     smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2])))
    #     output = multi_task_predict(bart, 'sentence_classification_head', smile)
    #     # output = np.array(output)
    #     # output = np.transpose(output)
    #     target.reshape((target.shape[0],1))
    #     trgt_list = [target[j].item() for j in range(len(target))]
    #     target = target.item()
    #     y_pred.append(output[0][0][1].exp().item())
    #     y.append(target - 4)


else:
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

        

    else: 
        print("ROC_AUC_SCORE: ", roc_auc_score(y, y_pred))
        y_pred_binary = np.array(y_pred) > 0.5
        print(classification_report(y, y_pred_binary))
        print("Confusion matrix:")
        print(confusion_matrix(y, y_pred_binary))
else:
    print(mean_squared_error([(ma -mi)*x + mi  for x in y] ,  [(ma -mi)*x +mi  for x in y_pred]))
