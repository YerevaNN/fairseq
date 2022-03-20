
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from ast import arg
import numpy as np
import deepchem as dc
import pandas as pd
import pandas as pd
import argparse
import os

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)

p.add_argument("--input", help="input file", type=str)
p.add_argument("--dataset-name", type=str, default="delaney")
p.add_argument("--smile-index", type=int, default=-1)
p.add_argument("--class-index", type=int)
p.add_argument("--task-count", type=int, default=1)

p.add_argument("--delimiter", type=str, default=",")
p.add_argument("--task-type", type=str, default='classification')

p.add_argument("--split-type", type=str, default='random')
p.add_argument("--filter", type=bool, default=False)
args = p.parse_args()


np.random.seed(123)

v = eval(f"dc.molnet.load_{args.dataset_name}")
tasks, datasets, transformers = v(splitter=args.split_type,
                                        featurizer = 'ECFP')
train_data, valid_data, test_data = datasets


train_df = train_data.to_dataframe()
valid_df = valid_data.to_dataframe() 
test_df = test_data.to_dataframe() 

os.system(f'mkdir {args.input}/{args.dataset_name}-bin')
os.system(f'mkdir {args.input}/{args.dataset_name}-bin/{args.dataset_name}')

#Remove some not usefull comlumns
remove_cols = [col for col in valid_df.columns if 'X' in col]
valid_df.drop(remove_cols, axis='columns', inplace=True)
train_df.drop(remove_cols, axis='columns', inplace=True)
test_df.drop(remove_cols, axis='columns', inplace=True)
remove_cols = [col for col in valid_df.columns if 'w' in col]
valid_df.drop(remove_cols, axis='columns', inplace=True)
train_df.drop(remove_cols, axis='columns', inplace=True)
test_df.drop(remove_cols, axis='columns', inplace=True)

test_df.to_csv(f"{args.input}/{args.dataset_name}-bin/{args.dataset_name}/test_{args.dataset_name}.csv")

train_df.to_csv(f"{args.input}/{args.dataset_name}-bin/{args.dataset_name}/train_{args.dataset_name}.csv")

valid_df.to_csv(f"{args.input}/{args.dataset_name}-bin/{args.dataset_name}/valid_{args.dataset_name}.csv")


loc = valid_df.columns.get_loc

# print(train_df.size)
# if args.filter:
#     nan_value = float("NaN")
#     train_df.replace("", nan_value, inplace=True)
#     train_df.dropna(subset=[train_df.iloc('y'), train_df.columns[args.smile_index]], inplace=True)
#     test_df.replace("", nan_value, inplace=True)
#     test_df.dropna(subset=[test_df.iloc('y'), test_df.columns[args.smile_index]], inplace=True)
#     valid_df.replace("", nan_value, inplace=True)
#     valid_df.dropna(subset=[valid_df[.iloc('y')], valid_df.columns[args.smile_index]], inplace=True)
# print(train_df.size)



if args.task_type == "regression":
    mi = min(list(map(int, train_df.iloc[:, args.class_index].tolist())))
    ma = max(list(map(int, train_df.iloc[:, args.class_index].tolist())))

    for i in [list(map(int, test_df.iloc[:, args.class_index].tolist())), list(map(int, test_df.iloc[:, args.class_index].tolist()))]:
        if min(i) < mi:
            mi = min(i)
        if max(i) > ma:
            ma = max(i)
    class_train = [(i - mi)/(ma - mi) for i in list(map(int, train_df.iloc[:, args.class_index].tolist()))] 
    class_test = [(i - mi)/(ma - mi) for i in list(map(int, test_df.iloc[:, args.class_index].tolist()))]
    class_val = [(i - mi)/(ma - mi) for i in list(map(int, valid_df.iloc[:, args.class_index].tolist()))]
    smiles_train = list(map(str, train_df.iloc[:, args.smile_index].tolist()))
    smiles_train = list(map(str, train_df.iloc[:, args.smile_index].tolist()))
    smiles_test = list(map(str, test_df.iloc[:, args.smile_index].tolist()))
    smiles_val = list(map(str, valid_df.iloc[:, args.smile_index].tolist()))
    print(f"Scale {args.task_type} dataset [0,1] interval")
    print(f"Regression task target minimum value: {mi} and max value: {ma} ")

if args.task_type == "classification":
    if args.task_count >1:
        class_train = list(train_df.iloc[:, loc('y1'):loc('y' + str(args.task_count))+1].values.tolist())
        class_test = list(test_df.iloc[:, loc('y1'):loc('y' + str(args.task_count))+1].values.tolist())
        class_val = list(valid_df.iloc[:, loc('y1'):loc('y' + str(args.task_count))+1].values.tolist())
    else:
        class_train = list(train_df.iloc[:, loc('y')].values.tolist())
        class_test = list(test_df.iloc[:, loc('y')].values.tolist())
        class_val = list(valid_df.iloc[:, loc('y')].values.tolist())

    smiles_val = list(map(str, valid_df.iloc[:, args.smile_index].tolist()))
    smiles_test = list(map(str, test_df.iloc[:, args.smile_index].tolist()))
    smiles_train = list(map(str, train_df.iloc[:, args.smile_index].tolist()))


if args.task_type =='regression':
    os.system(f'mkdir {args.input}/{args.dataset_name}-bin/label')
    with open(f"{args.input}/{args.dataset_name}-bin/label/train.label", "w") as f:
        for item in class_train:
            f.write("%s\n" % item)
    with open(f"{args.input}/{args.dataset_name}-bin/label/valid.label", "w") as f:
        for item in class_val:
            f.write("%s\n" % item)
    with open(f"{args.input}/{args.dataset_name}-bin/label/test.label", "w") as f:
        for item in class_val:
            f.write("%s\n" % item)

os.system(f"mkdir {args.input}/{args.dataset_name}-bin/raw")
os.system(f"mkdir {args.input}/{args.dataset_name}-bin/tokenized")
os.system(f"mkdir {args.input}/{args.dataset_name}-bin/processed")

os.system(f'mkdir {args.input}/{args.dataset_name}-bin/processed/label')
with open(f"{args.input}/{args.dataset_name}-bin/processed/label/train.label", "w") as f:
    for item in class_train:
        f.write("%s\n" % item)
with open(f"{args.input}/{args.dataset_name}-bin/processed/label/valid.label", "w") as f:
    for item in class_val:
        f.write("%s\n" % item)
with open(f"{args.input}/{args.dataset_name}-bin/processed/label/test.label", "w") as f:
    for item in class_val:
        f.write("%s\n" % item)



print(f"{args.dataset_name} Train Length: {len(smiles_train)}")
print(f"{args.dataset_name} Valid Length: {len(smiles_val)}")
print(f"{args.dataset_name} Test Length: {len(smiles_test)}")

names = ["train", "valid", "test"]
X_splits = []
y_splits = []



# Write Raw Splits
print("Writing Input Splits")
for name, smiles in zip(names, (smiles_train, smiles_val, smiles_test)):
    print(name + ".target")
    print(args.dataset_name )
    new_path = f"{args.input}/{args.dataset_name}-bin/raw/" + name + ".input" 

    X_splits.append(new_path)
    with open(new_path, "w+") as f:
        for smile in smiles:
            f.write(f"{smile}\n")

print("Writing Output Splits")
for name, targets in zip(names, (class_train, class_val, class_test)):
    print( f"{args.input}/{args.dataset_name}-bin/raw/" + name + ".target")
    print(args.dataset_name )
    new_path = f"{args.input}/{args.dataset_name}-bin/raw/" + name + ".target"
    y_splits.append(new_path)
    with open(new_path, "w+") as f:
        for target in targets:
            f.write(f"{str(target)}\n")

# Tokenize Texts
print("Tokenizing")
splits = []
for path in X_splits:
    cur_path = path.replace('raw', 'tokenized')
    print(path)
    print(cur_path)
    splits.append(cur_path)
    os.system(
        f"python {args.input}/fairseq/scripts/spm_parallel.py --input {path} --outputs {cur_path} --model /home/gayane/BartLM/Bart/chemical/checkpoints/chemical/tokenizer/chem.model")

X_splits = splits

os.system(('fairseq-preprocess --only-source '
           f'--trainpref "{X_splits[0]}" '
           f'--validpref "{X_splits[1]}" '
           f'--testpref "{X_splits[2]}" '
           f'--destdir "{args.input}/{args.dataset_name}-bin/processed/input0" --workers 60 '
           '--srcdict /home/gayane/BartLM/Bart/chemical/checkpoints/chemical/tokenizer/chem.vocab.fs'))
if args.task_type == "classification":
    os.system(('fairseq-preprocess '
           '--only-source '
           f'--trainpref "{y_splits[0]}" '
           f'--validpref "{y_splits[1]}" '
           f'--testpref "{y_splits[2]}" '
           f'--destdir "{args.input}/{args.dataset_name}-bin/processed/label" --workers 60 '))

    