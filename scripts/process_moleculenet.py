from scripts import *
from scripts import process
import numpy as np
from ast import arg
from rdkit import Chem
import deepchem as dc
import pandas as pd
import argparse
import json
import os
# os.environ['MKL_THREADING_LAYER'] = 'GNU'

EMPTY_INDEX = -1

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)

# p.add_argument("--input", help="input file", type=str, required=True)
p.add_argument("--dataset-name", type=str, required=True)
p.add_argument("--delimiter", type=str, default=",")

args = p.parse_args()

np.random.seed(123)

with open('/home/gayane/BartLM/fairseq/scripts/datasets.json') as f:
    datasets_json = json.load(f)

dataset = datasets_json[args.dataset_name]
si = dataset['smiles_index']
input_path = "/home/gayane/BartLM"
store_path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data"

if len(dataset["class_index"]) > 1:
    path = list()
    for i in range(len(dataset["class_index"])):
        os.system(f'mkdir -p {store_path}/{args.dataset_name}_{i}')
        os.system(f'mkdir -p {store_path}/{args.dataset_name}_{i}/{args.dataset_name}')
        path.append(f"{store_path}/{args.dataset_name}_{i}/{args.dataset_name}/")

else:
    os.system(f'mkdir -p {store_path}/{args.dataset_name}')
    os.system(f'mkdir -p {store_path}/{args.dataset_name}/{args.dataset_name}')
    path = f"{store_path}/{args.dataset_name}/{args.dataset_name}/"


if args.dataset_name == "regAcuteOralToxicity":
    df = process.regAcuteOralToxicity(args.dataset_name, path)

elif  args.dataset_name == "classAcuteOralToxicity":
    df = process.classAcuteOralToxicity(args.dataset_name, path)


elif args.dataset_name == "Genotoxicity":
    df = process.genotoxicity(args.dataset_name, path)

elif args.dataset_name == "japan":
    df = process.japan(args.dataset_name, path) 

elif args.dataset_name == "snyder_negatives_451":
    df = process.snyder_negatives_451(args.dataset_name, path)   
    
elif args.dataset_name == "Ames":
    df = process.ames(args.dataset_name, path, dataset["smiles_col_name"], dataset['label_col_name'])

elif args.dataset_name == "ZINC":
    train_df, valid_df, test_df = process.ZINC(args.dataset_name, path)
elif args.dataset_name.startswith("ZINC_part"):
    train_df, valid_df, test_df = process.ZINC_by_parts(args.dataset_name, path)
elif args.dataset_name == "USPTO-50k":
    train_df, valid_df, test_df = process.USPTO(args.dataset_name, "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/USPTO-50k/")
elif args.dataset_name == "BBBP-balanced":
    test_df = pd.read_csv(f"{store_path}/{args.dataset_name}/{args.dataset_name}/test_{args.dataset_name}.csv")
    train_df = pd.read_csv(f"{store_path}/{args.dataset_name}/{args.dataset_name}/train_{args.dataset_name}.csv")
    valid_df = pd.read_csv(f"{store_path}/{args.dataset_name}/{args.dataset_name}/valid_{args.dataset_name}.csv")

elif args.dataset_name.startswith("B_"):
    train_df = process.scaff(args.dataset_name, "train")
    valid_df = train_df
    test_df = valid_df

else:
    # For MoleculeNet data
    
    v = eval(f"dc.molnet.load_{dataset['load_name']}")
    tasks, datasets, transformers = v(splitter=dataset['split_type'],
                                                featurizer = 'ECFP')

    train_data, valid_data, test_data = datasets


    train_df = train_data.to_dataframe()
    valid_df = valid_data.to_dataframe() 
    test_df = test_data.to_dataframe() 


    # Remove some not usefull comlumns
    remove_cols = [col for col in valid_df.columns if 'X' in col]
    valid_df.drop(remove_cols, axis='columns', inplace=True)
    train_df.drop(remove_cols, axis='columns', inplace=True)
    test_df.drop(remove_cols, axis='columns', inplace=True)
    remove_cols = [col for col in valid_df.columns if 'w' in col]
    valid_df.drop(remove_cols, axis='columns', inplace=True)
    train_df.drop(remove_cols, axis='columns', inplace=True)
    test_df.drop(remove_cols, axis='columns', inplace=True)
    
    # print(train_df.head())

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
    else:
        train_df.replace("", EMPTY_INDEX, inplace=True)
        train_df.fillna(EMPTY_INDEX, inplace=True)
        valid_df.replace("", EMPTY_INDEX, inplace=True)
        valid_df.fillna(EMPTY_INDEX, inplace=True)
        test_df.replace("", EMPTY_INDEX, inplace=True)
        test_df.fillna(EMPTY_INDEX, inplace=True)

if (args.dataset_name=="japan" or args.dataset_name=="snyder_negatives_451"
    or args.dataset_name == "Ames" or args.dataset_name == "classAcuteOralToxicity" 
    or args.dataset_name == "regAcuteOralToxicity"):
    label_col_name = dataset["label_col_name"]
    smiles_col_name = dataset['smiles_col_name']
    
    train_df, valid_df, test_df = process.split_train_val_test(df, dataset['smiles_col_name'], dataset['label_col_name'])
if args.dataset_name == "Genotoxicity":
    list_df, test_df = process.cross_val(df,  dataset['smiles_col_name'], dataset['label_col_name'], k_fold=5)
if len(dataset["class_index"]) >1:
    print("___________________")
    for i in range(len(dataset["class_index"])):
        test_df.to_csv(f"{store_path}/{args.dataset_name}_{i}/{args.dataset_name}/test_{args.dataset_name}.csv")
        train_df.to_csv(f"{store_path}/{args.dataset_name}_{i}/{args.dataset_name}/train_{args.dataset_name}.csv")
        valid_df.to_csv(f"{store_path}/{args.dataset_name}_{i}/{args.dataset_name}/valid_{args.dataset_name}.csv")
else:

    test_df.to_csv(f"{store_path}/{args.dataset_name}/{args.dataset_name}/test_{args.dataset_name}.csv")
    train_df.to_csv(f"{store_path}/{args.dataset_name}/{args.dataset_name}/train_{args.dataset_name}.csv")
    valid_df.to_csv(f"{store_path}/{args.dataset_name}/{args.dataset_name}/valid_{args.dataset_name}.csv")


loc = valid_df.columns.get_loc


if dataset["type"] == "regression":
    print(len(dataset['class_index']))
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
        print("___________________")
        class_dict = {}
        for cii in range(len(dataset["class_index"])):
            class_dict["class_train" +str(cii)] = list(map(int, train_df.iloc[:, dataset["class_index"][cii]].tolist()))
            class_dict["class_val" +str(cii)] =list(map(int, valid_df.iloc[:, dataset["class_index"][cii]].tolist()))
            class_dict["class_test" +str(cii)] =list(map(int, test_df.iloc[:, dataset["class_index"][cii]].tolist()))

    else:
        ci = dataset['class_index'][0]
        class_train = list(map(int, train_df.iloc[:, ci].tolist()))
        class_val = list(map(int, valid_df.iloc[:, ci].tolist()))
        class_test = list(map(int, test_df.iloc[:, ci].tolist()))

# train_df.iloc[:, si] = train_df.iloc[:, si].apply(Chem.CanonSmiles)
# valid_df.iloc[:, si] = valid_df.iloc[:, si].apply(Chem.CanonSmiles)
# test_df.iloc[:, si] = test_df.iloc[:, si].apply(Chem.CanonSmiles)

smiles_train = list(map(str, train_df.iloc[:, si].tolist()))
smiles_val = list(map(str, valid_df.iloc[:, si].tolist()))
smiles_test = list(map(str, test_df.iloc[:, si].tolist()))


if dataset["type"] =='regression':
    os.system(f'mkdir {store_path}/{args.dataset_name}/label')
    with open(f"{store_path}/{args.dataset_name}/label/train.label", "w") as f:
        for item in class_train:
            f.write("%s\n" % item)
    with open(f"{store_path}/{args.dataset_name}/label/valid.label", "w") as f:
        for item in class_val:
            f.write("%s\n" % item)
    with open(f"{store_path}/{args.dataset_name}/label/test.label", "w") as f:
        for item in class_test:
            f.write("%s\n" % item)


if len(dataset['class_index'])>1:
    for i in range(len(dataset['class_index'])):


        os.system(f"mkdir {store_path}/{args.dataset_name}_{i}/raw")
        os.system(f"mkdir {store_path}/{args.dataset_name}_{i}/tokenized")
        os.system(f"mkdir {store_path}/{args.dataset_name}_{i}/processed")
        
        os.system(f'mkdir {store_path}/{args.dataset_name}_{i}/processed/label')
        with open(f"{store_path}/{args.dataset_name}_{i}/processed/label/train.label", "w") as f:
            for item in  class_dict["class_train" +str(i)]:
                f.write("%s\n" % item)
        with open(f"{store_path}/{args.dataset_name}_{i}/processed/label/valid.label", "w") as f:
            for item in class_dict["class_val" +str(i)]:
                f.write("%s\n" % item)
        with open(f"{store_path}/{args.dataset_name}_{i}/processed/label/test.label", "w") as f:
            for item in class_dict["class_test" +str(i)]:
                f.write("%s\n" % item)
    
else:
    os.system(f"mkdir {store_path}/{args.dataset_name}/raw")
    os.system(f"mkdir {store_path}/{args.dataset_name}/tokenized")
    os.system(f"mkdir {store_path}/{args.dataset_name}/processed")
    os.system(f'mkdir {store_path}/{args.dataset_name}/processed/label')
    os.system(f"mkdir {store_path}/{args.dataset_name}/processed/input0")
    with open(f"{store_path}/{args.dataset_name}/processed/label/train.label", "w") as f:
        for item in class_train:
            f.write("%s\n" % item)
    with open(f"{store_path}/{args.dataset_name}/processed/label/valid.label", "w") as f:
        for item in class_val:
            f.write("%s\n" % item)
    with open(f"{store_path}/{args.dataset_name}/processed/label/test.label", "w") as f:
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
    if len(dataset["class_index"]) >1:
        print("________________________")
        path_ = []
        for i in range(len(dataset["class_index"])):
            new_path = f"{store_path}/{args.dataset_name}_{i}/raw/" + name + ".input"
            path_.append(new_path)
            with open(new_path, "w+") as f:
                for smile in smiles:
                    f.write(f"{smile}\n")
        X_splits.append(path_)
    else:
        new_path = f"{store_path}/{args.dataset_name}/raw/" + name + ".input" 
        X_splits.append(new_path)
        with open(new_path, "w+") as f:
            for smile in smiles:
                f.write(f"{smile}\n")

print("Writing Output Splits")
if len(dataset['class_index'])>1:
    # pass
    for i in range(len(dataset['class_index'])):
        y_splits_current = []
        for name, targets in zip(names, (class_dict["class_train" +str(i)], class_dict["class_val" +str(i)], class_dict["class_test" +str(i)])):
            print(args.dataset_name )
            new_path = f"{store_path}/{args.dataset_name}_{i}/raw/" + name + ".target"
            print(new_path)
            y_splits_current.append(new_path)
            with open(new_path, "w+") as f:
                for target in targets:
                    f.write(f"{str(target)}\n")
        y_splits.append(y_splits_current)
    
else:
    for name, targets in zip(names, (class_train, class_val, class_test)):
        print(args.dataset_name )
        new_path = f"{store_path}/{args.dataset_name}/raw/" + name + ".target"
        print(new_path)
        y_splits.append(new_path)
        with open(new_path, "w+") as f:
            for target in targets:
                f.write(f"{str(target)}\n")

# Tokenize Texts

print("Tokenizing")

if len(dataset["class_index"]) > 1:

    split_train = []
    split_valid = []
    split_test = []
    for train_path, valid_path, test_path in zip(X_splits[0], X_splits[1], X_splits[2]):
        cur_train = train_path.replace('raw', 'tokenized')
        cur_valid = valid_path.replace('raw', 'tokenized')
        cur_test = test_path.replace('raw', 'tokenized')
        # print(path)
        # print(cur_path)
        split_train.append(cur_train)
        split_valid.append(cur_valid)
        split_test.append(cur_test)
        cmd = f"python {input_path}/fairseq/scripts/spm_parallel.py --input {train_path} --outputs {cur_train} --model /home/gayane/BartLM/Bart/chemical/tokenizer/chem.model"
        print(cmd)
        os.system(cmd)
        cmd = f"python {input_path}/fairseq/scripts/spm_parallel.py --input {valid_path} --outputs {cur_valid} --model /home/gayane/BartLM/Bart/chemical/tokenizer/chem.model"
        print(cmd)
        os.system(cmd)
        cmd = f"python {input_path}/fairseq/scripts/spm_parallel.py --input {test_path} --outputs {cur_test} --model /home/gayane/BartLM/Bart/chemical/tokenizer/chem.model"
        print(cmd)
        os.system(cmd)
    X_splits[0], X_splits[1], X_splits[2] = split_train, split_valid, split_test

else:
    splits = []
    for path in X_splits:
        cur_path = path.replace('raw', 'tokenized')
        print(path)
        print(cur_path)
        splits.append(cur_path)
        cmd = f"python {input_path}/fairseq/scripts/spm_parallel.py --input {path} --outputs {cur_path} --model /home/gayane/BartLM/Bart/chemical/tokenizer/chem.model"
        print(cmd)
        os.system(cmd)

    X_splits = splits

if dataset["type"] == "classification":
    if len(dataset["class_index"]) > 1:
        for i in range(len(dataset['class_index'])):

            os.system(('fairseq-preprocess --only-source '
                f'--trainpref "{X_splits[0][i]}" '
                f'--validpref "{X_splits[1][i]}" '
                f'--testpref "{X_splits[2][i]}" '
                f'--destdir "{store_path}/{args.dataset_name}_{i}/processed/input0" --workers 60 '
                '--srcdict /home/gayane/BartLM/Bart/chemical/tokenizer/chem.vocab.fs'))
            os.system(('fairseq-preprocess '
                '--only-source '
                f'--trainpref "{y_splits[i][0]}" '
                f'--validpref "{y_splits[i][1]}" '
                f'--testpref "{y_splits[i][2]}" '
                f'--destdir "{store_path}/{args.dataset_name}_{i}/processed/label" --workers 60 '))

    
    else:

        os.system(('fairseq-preprocess --only-source '
            f'--trainpref "{X_splits[0]}" '
            f'--validpref "{X_splits[1]}" '
            f'--testpref "{X_splits[2]}" '
            f'--destdir "{store_path}/{args.dataset_name}/processed/input0" --workers 60 '
            '--srcdict /home/gayane/BartLM/Bart/chemical/tokenizer/chem.vocab.fs'))
        os.system(('fairseq-preprocess '
            '--only-source '
            f'--trainpref "{y_splits[0]}" '
            f'--validpref "{y_splits[1]}" '
            f'--testpref "{y_splits[2]}" '
            f'--destdir "{store_path}/{args.dataset_name}/processed/label" --workers 60 '))
