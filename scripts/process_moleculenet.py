
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from ast import arg
import numpy as np
import deepchem as dc
import pandas as pd
import argparse
import json
import os

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)

p.add_argument("--input", help="input file", type=str)
p.add_argument("--dataset-name", type=str, default="delaney")
# p.add_argument("--smile-index", type=int, default=-1)
# p.add_argument("--class-index",  type=int, default=-2)
# p.add_argument("--task-count", type=int, default=1)
# p.add_argument("")
p.add_argument("--delimiter", type=str, default=",")
# p.add_argument("--task-type", type=str, default='classification')

# p.add_argument("--split-type", type=str, default='random')
# p.add_argument("--filter", type=bool, default=False)
args = p.parse_args()

np.random.seed(123)

with open('/home/gayane/BartLM/fairseq/scripts/datasets.json') as f:
    datasets_json = json.load(f)

dataset = datasets_json[args.dataset_name]

v = eval(f"dc.molnet.load_{dataset['load_name']}")
tasks, datasets, transformers = v(splitter=dataset['split_type'],
                                            featurizer = 'ECFP')

train_data, valid_data, test_data = datasets


train_df = train_data.to_dataframe()
valid_df = valid_data.to_dataframe() 
test_df = test_data.to_dataframe() 

os.system(f'mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}')
os.system(f'mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/{args.dataset_name}')

# Remove some not usefull comlumns
remove_cols = [col for col in valid_df.columns if 'X' in col]
valid_df.drop(remove_cols, axis='columns', inplace=True)
train_df.drop(remove_cols, axis='columns', inplace=True)
test_df.drop(remove_cols, axis='columns', inplace=True)
remove_cols = [col for col in valid_df.columns if 'w' in col]
valid_df.drop(remove_cols, axis='columns', inplace=True)
train_df.drop(remove_cols, axis='columns', inplace=True)
test_df.drop(remove_cols, axis='columns', inplace=True)

print(train_df.head())
test_df.to_csv(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/{args.dataset_name}/test_{args.dataset_name}.csv")

train_df.to_csv(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/{args.dataset_name}/train_{args.dataset_name}.csv")

valid_df.to_csv(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/{args.dataset_name}/valid_{args.dataset_name}.csv")

si = dataset['smiles_index']

if dataset['filter']:
    assert len(dataset['class_index']) == 1, "We do not want to filter multi-task datasets."
    ci = dataset['class_index'][0]

    nan_value = float("NaN")
    train_df.replace("", nan_value, inplace=True)
    train_df.dropna(subset=[train_df.columns[ci], train_df.columns[si]], inplace=True)

    valid_df.replace("", nan_value, inplace=True)
    valid_df.dropna(subset=[valid_df.columns[ci], valid_df.columns[si]], inplace=True)

    test_df.replace("", nan_value, inplace=True)
    test_df.dropna(subset=[test_df.columns[ci], test_df.columns[si]], inplace=True)


loc = valid_df.columns.get_loc


if dataset["type"] == "regression":
    assert len(dataset['class_index']) == 1, "Regression tasks are always single-task."
    ci = dataset['class_index'][0]
    mi = dataset['minimum']
    ma = dataset['maximum']

    class_train = [(i - mi)/(ma - mi) for i in list(train_df.iloc[:, ci].values.tolist())] 
    class_test = [(i - mi)/(ma - mi) for i in list(test_df.iloc[:, ci].values.tolist())]
    class_val = [(i - mi)/(ma - mi) for i in list(valid_df.iloc[:, ci].values.tolist())]
    
    print(f"Scale {args.dataset_name} (type={dataset['type']}) dataset [0,1] interval")
    print(f"Regression task target minimum value: {mi} and max value: {ma} ")

if dataset["type"] == "classification":
    if len(dataset["class_index"]) >1:
        class_dict = {}
        for cii in range(1, len(dataset["class_index"])):
            class_dict["class_train" +str(cii)] =list(map(int, train_df.iloc[:, dataset["class_index"][cii]].tolist()))
            class_dict["class_val" +str(cii)] =list(map(int, valid_df.iloc[:, dataset["class_index"][cii]].tolist()))
            class_dict["class_test" +str(cii)] =list(map(int, test_df.iloc[:, dataset["class_index"][cii]].tolist()))

    else:
        ci = dataset['class_index'][0]
        class_train = list(map(int, train_df.iloc[:, ci].tolist()))
        class_val = list(map(int, valid_df.iloc[:, ci].tolist()))
        class_test = list(map(int, test_df.iloc[:, ci].tolist()))

smiles_train = list(map(str, train_df.iloc[:, si].tolist()))
smiles_val = list(map(str, valid_df.iloc[:, si].tolist()))
smiles_test = list(map(str, test_df.iloc[:, si].tolist()))

if dataset["type"] =='regression':
    os.system(f'mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/label')
    with open(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/label/train.label", "w") as f:
        for item in class_train:
            f.write("%s\n" % item)
    with open(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/label/valid.label", "w") as f:
        for item in class_val:
            f.write("%s\n" % item)
    with open(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/label/test.label", "w") as f:
        for item in class_test:
            f.write("%s\n" % item)

os.system(f"mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/raw")
os.system(f"mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/tokenized")
os.system(f"mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed")
if len(dataset['class_index'])>1:
    for i in range(1, len(dataset['class_index'])):
        
        os.system(f'mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/label{i}')
        with open(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/label{i}/train.label", "w") as f:
            for item in  class_dict["class_train" +str(i)]:
                f.write("%s\n" % item)
        with open(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/label{i}/valid.label", "w") as f:
            for item in class_dict["class_val" +str(i)]:
                f.write("%s\n" % item)
        with open(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/label{i}/test.label", "w") as f:
            for item in class_dict["class_test" +str(i)]:
                f.write("%s\n" % item)
    
else:
    os.system(f'mkdir /home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/label')
    with open(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/label/train.label", "w") as f:
        for item in class_train:
            f.write("%s\n" % item)
    with open(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/label/valid.label", "w") as f:
        for item in class_val:
            f.write("%s\n" % item)
    with open(f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/label/test.label", "w") as f:
        for item in class_test:
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
    new_path = f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/raw/" + name + ".input" 

    X_splits.append(new_path)
    with open(new_path, "w+") as f:
        for smile in smiles:
            f.write(f"{smile}\n")

print("Writing Output Splits")
if len(dataset['class_index'])>1:
    pass
    # for i in range(1, len(dataset['class_index'])):
    #     for name, targets in zip(names, (class_dict["class_train" +str(i)], class_dict["class_val" +str(i)], class_dict["class_test" +str(i)])):
    #         print( f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/raw/{i}" + name + ".target")
    #         print(args.dataset_name )
    #         new_path = f"/home/gayane/BartLM/Bart/chemyical/evaluation_data/{args.dataset_name}/raw/{i}" + name + ".target"
    #         y_splits.append(new_path)
    #         with open(new_path, "w+") as f:
    #             for target in targets:
    #                 f.write(f"{str(target)}\n")
else:
    for name, targets in zip(names, (class_train, class_val, class_test)):
        print( f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/raw/" + name + ".target")
        print(args.dataset_name )
        new_path = f"/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/raw/" + name + ".target"
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
        f"python {args.input}/fairseq/scripts/spm_parallel.py --input {path} --outputs {cur_path} --model /home/gayane/BartLM/Bart/chemical/tokenizer/chem.model")

X_splits = splits

os.system(('fairseq-preprocess --only-source '
           f'--trainpref "{X_splits[0]}" '
           f'--validpref "{X_splits[1]}" '
           f'--testpref "{X_splits[2]}" '
           f'--destdir "/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/input0" --workers 60 '
           '--srcdict /home/gayane/BartLM/Bart/chemical/tokenizer/chem.vocab.fs'))
if dataset["type"] == "classification":
    if len(dataset["class_index"]) >1:
        for i in range(1, len(dataset['class_index'])):
            os.system(('fairseq-preprocess '
            '--only-source '
            f'--trainpref "{class_dict["class_train" +str(i)][0]}" '
            f'--validpref "{class_dict["class_val" +str(i)][1]}" '
            f'--testpref "{class_dict["class_test" +str(i)][2]}" '
            f'--destdir "/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/label{i}" --workers 60 '))

    
    else:
        os.system(('fairseq-preprocess '
            '--only-source '
            f'--trainpref "{y_splits[0]}" '
            f'--validpref "{y_splits[1]}" '
            f'--testpref "{y_splits[2]}" '
            f'--destdir "/home/gayane/BartLM/Bart/chemical/evaluation_data/{args.dataset_name}/processed/label" --workers 60 '))

    