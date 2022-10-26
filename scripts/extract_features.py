from fairseq.data.data_utils import load_indexed_dataset
from fairseq.data.data_utils import collate_tokens
from fairseq.models.bart import BARTModel
from fairseq.data import Dictionary
import torch.nn.functional as F 
from tqdm import tqdm
import numpy as np
import argparse
import torch
import os



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--output_path', type=str, default='/mnt/good/gayane/data/data_load_folder')
args = parser.parse_args()


dataset = args.dataset
np_filename = os.path.join(args.output_path, f"np_{dataset}.npy")
print(np_filename)

if os.path.exists(np_filename):
    print(f"The file {np_filename} already exists")
    exit()
    

store_path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data"
model = f"{store_path}/{dataset}/processed"

chkpt_path = '/home/gayane/BartLM/checkpoints/checkpoint_last.pt'
print(chkpt_path)  # BACE_bs_16_lr_3e-5_totalNum_1135_warmup_181/ in test 

bart = BARTModel.from_pretrained(model,  checkpoint_file = chkpt_path, 
                                bpe="sentencepiece",
                                sentencepiece_model="/home/gayane/BartLM/Bart/chemical/tokenizer/chem.model")

input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")

bart.eval()
bart.cuda()


data_type = 'train'
def get_data(data_type):
    input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")
    smiles = load_indexed_dataset(
        f"{store_path}/{dataset}/processed/input0/{data_type}", input_dict)
    return list(smiles)

sm_train = get_data("train") # , y_pred_train
sm_valid = get_data("valid") # , y_pred_valid
sm_test = get_data("test") # , y_pred_test


smi = []

def get_features(sm):
    umap_X = []
    with torch.no_grad():
        count = len(sm)
        batch_count = int(np.ceil(count / args.batch_size))
        for i in tqdm(range(batch_count)):
            inputs = sm[i * args.batch_size : (i+1) * args.batch_size]
            batch = collate_tokens(
                inputs, pad_idx=1
            ).to(bart.device)[:, :128] # manually cropping to the max length

            last_layer_features = bart.extract_features(batch)
            
            assert len(inputs) == len(last_layer_features), "len(inputs) == len(last_layer_features)"
    
            for inp, feat in zip(inputs, last_layer_features.to("cpu")):
#                 print(inp.shape, feat[:len(inp)].mean(axis=0).shape)
                # manually cropping till the padding
                umap_X.append(feat[:len(inp)].mean(axis=0).numpy())
    return umap_X

print("starting extract train data")
umap_X_train = get_features(sm_train)
print("starting extract validation data")
umap_X_valid = get_features(sm_valid)
print("starting extract test data")
umap_X_test = get_features(sm_test)


# umap_X_train = [i.numpy() for i in umap_X_train]
# umap_X_valid = [i.numpy() for i in umap_X_valid]
# umap_X_test = [i.numpy() for i in umap_X_test]


X = np.array(umap_X_train + umap_X_valid + umap_X_test)
print("X.shape", X.shape)

print(f"Saving to {np_filename}")
np.save(np_filename, X)
    