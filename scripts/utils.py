from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, classification_report, mean_squared_error, confusion_matrix
import pandas as pd
def compute_rmse(y_pred, y, ma, mi):
    y_prd = [(ma -mi)*x +mi  for x in y_pred]
    y_l = [(ma -mi)*x + mi  for x in y]
    df = pd.DataFrame(data={"y_l": y, "y_pred": y_prd, "y_l_scale": y_l, "y_pred_scale": y_pred})
    print(mean_squared_error([(ma -mi)*x + mi  for x in y], [(ma -mi)*x +mi  for x in y_pred]))

def compute_auc(y_pred, y):
    print("ROC_AUC_SCORE: ", roc_auc_score(y, y_pred))

def compute_prc_auc(y_pred, y):
    prc_auc_list = list()
    precision, recall, thresholds = precision_recall_curve(y, y_pred)
    prc_auc_list.append(auc(recall, precision))
    print("PRC_AUC_SCORE: ", auc(recall, precision))

def compute_conf_matrix(y_pred, y):
    print("Confusion matrix:")
    y_pred_binary = np.array(y_pred) > 0.5
    print(classification_report(y, y_pred_binary))

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
