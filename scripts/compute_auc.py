from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, classification_report, mean_squared_error, confusion_matrix
from fairseq.data.data_utils import load_indexed_dataset
# from sklearn.linear_model import ridge_regression
from fairseq.tasks.multi_input_sentence_prediction import OurDataset
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
os.environ['MKL_THREADING_LAYER'] = 'GNU'


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
parser.add_argument('--subtask',
                        help='subtask')
parser.add_argument('--warmup-update',
                        help='warmup update', default=46)
parser.add_argument('--total-number-update',
                        help='total number update', default=290)
parser.add_argument('--lr', default="5e-6",
                        help='learning rate')
parser.add_argument('--dropout', default="0.3",
                        help='dropout')
parser.add_argument('--r3f',
                        help='lambda param')
parser.add_argument('--noise_type',
                        help='normal or unniform')
parser.add_argument('--inp-count', default=1,
                        help='input count')
parser.add_argument('--checkpoint_name', default="checkpoint_best.pt")
args = parser.parse_args()

dataset = args.dataset_name #if args.dataset_name in set(["esol", "freesolv", "lipo", "Ames", "BBBP", "BACE", "HIV"]) else f"{args.dataset_name}_{args.subtask}"

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

warmup = args.warmup_update
totNumUpdate = args.total_number_update
lr = args.lr
noise_type = args.noise_type
r3f_lambda = args.r3f
drout = args.dropout
# _noise_type_uniform_r3f_lambda_0.7
noise_params = f"_noise_type_{noise_type}_r3f_lambda_{r3f_lambda}" if noise_type in ["uniform", "normal"] else ""

chkpt_path = f"/mnt/good/gayane/data/chkpt/{dataset}_bs_16_dropout_{drout}_lr_{lr}_totalNum_{totNumUpdate}_warmup_{warmup}{noise_params}/{args.checkpoint_name}"
# chkpt_path = "/mnt/good/gayane/data/chkpt/BBBP_bs_16_dropout_0.1_lr_3e-5_totalNum_1020_warmup_163_noise_type_uniform_r3f_lambda_1.0/checkpoint_best.pt"
print(chkpt_path)  # BACE_bs_16_lr_3e-5_totalNum_1135_warmup_181/ in test 
bart = BARTModel.from_pretrained(model,  checkpoint_file = chkpt_path, 
                                 bpe="sentencepiece",
                                 sentencepiece_model="/home/gayane/BartLM/Bart/chemical/tokenizer/chem.model")
bart.eval()
bart.cuda(device=1)


input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")
dataset_type = "valid"
smiles = list(load_indexed_dataset(
    f"{store_path}/{dataset}/processed/input0/{dataset_type}", input_dict))
if int(args.inp_count) > 1:
    input1_path = f"{store_path}/{dataset}/raw/{dataset_type}.input1"
    with open(input1_path) as f:
        lines = f.readlines()
        lines = [list(map(int,list(l.strip()))) for l in lines]
    fp_tokens = [torch.Tensor(item) for item in lines]
    # tns = torch.Tensor(lines)
    # input1 = OurDataset(lines)
    # fp_tokens = [torch.cat((sm, fp)) for sm, fp in zip(smiles,lines)]
    # smiles = inp
head_name = "sentence_classification_head"  # "multi_input_sentence_classification_head" if int(args.inp_count) > 1 else 

if len(dataset_js["class_index"])>1:
    test_label_path = list()
    for i in range(len(dataset_js["class_index"])):
        test_label_path.append(f"{store_path}/{dataset}_{i}/processed/label/{dataset_type}")

else:
    test_label_path = f"{store_path}/{dataset}/processed/label/{dataset_type}"

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
if len(dataset_js["class_index"])>1:
    y_pred_list = list()
    y_list = list()
    for j in range(len(dataset_js["class_index"])):
        y_pred = list()
        y = list()
        for i, (smile, target) in tqdm(list(enumerate(zip(smiles, targets_list[j])))):
            smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2])))
            output = bart.predict(f'{head_name}{j}', smile)
            target = target[0].item()
            y_pred.append(output[0][1].exp().item())
            y.append(target - 4)
        y_pred_list.append(y_pred)
        y_list.append(y)
else:
    for i, (smile, fp, target) in tqdm(list(enumerate(zip(smiles, fp_tokens, targets)))):
        smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2])))  
        if task_type =="classification":
            output = bart.predict(head_name, smile, fp)
            target = target[0].item()
            y_pred.append(output[0][1].exp().item())
            y.append(target - 4)
            sm.append(bart.decode(smile))
            
        elif task_type == "regression":
            output = bart.predict(head_name, smile, return_logits=True)
            y_pred.append(output[0][0].item())
            y.append(target)
    d = {"SMILES": sm, "prediction": y_pred , "y_true": y }
    df = pd.DataFrame(d) 
    df = df.dropna()
    df.to_csv(f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/{args.dataset_name}/{args.dataset_name}_test_.csv")     

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
