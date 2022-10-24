from scripts.process_moleculenet_new import X_splits
from sklearn.model_selection import train_test_split
from scripts.utils import fairseq_preprocess_cmd, tokenize, create_raw, generateMurcoScaffold, getMurcoScaffoldList
from sklearn.model_selection import KFold
from itertools import chain
import pandas as pd
import numpy as np
import json
import csv
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'



def scaff(dataset_name, type):

    path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/"
    file=open(path + f"{dataset_name}/{dataset_name}/{type}_{dataset_name}.csv")
    file = file.read().split('\n')
    sm = list()
    col = list()

    for i in range(1,len(file)-1):
        sm.append(file[i].split(',')[-1])
        col.append(file[i].split(',')[-2].strip("'"))
    bool_col = [float(i) for i in col]

    d = {"Classification": bool_col , "SMILES": sm}
    df = pd.DataFrame(d) 
    df = df.dropna()
    return df

def Genotoxicity(dataset_name, path):
    print (dataset_name)

    file=open(path + "Genotoxicity.txt")
    file = file.read().split('\n')
    sm = list()
    col = list()

    for i in range(1,len(file)):
        sm.append(file[i].split('\t')[0].strip('"'))
        col.append(file[i].split('\t')[1])
    bool_col = [int(i == "positive") for i in col]

    d = {"SMILES": sm, "Classification": bool_col }
    df = pd.DataFrame(d) 
    df = df.dropna()
    df.to_csv(path +"Genotoxicity.csv")

    return df


def Ames(dataset_name, path, smiles_col_name = "Canonical_Smiles", label_col_name = "Activity"):
    print(path)
    print (f"{path}{dataset_name}.csv")
    df = pd.read_csv(f"{path}{dataset_name}.csv")
    d = {"SMILES": df[smiles_col_name], "Classification": df[label_col_name] }
    df = pd.DataFrame(d) 
    # df = df.dropna()
    print(df.columns, df.shape)
    print(df['SMILES'].head())

    train_df, valid_df, test_df = split_train_val_test(df, smiles_col_name, label_col_name)
    return train_df, valid_df, test_df

def ZINC(dataset_name, path):
    print(path)
    path_ = f"{path}"
    dataset_name = "ZINC"
    print(f"{path_}{dataset_name}.csv")
    print(path)
    print (f"{path}{dataset_name}.csv")
    df_train = pd.read_csv(f"{path}train_{dataset_name}.csv")
    df_val = pd.read_csv(f"{path}valid_{dataset_name}.csv")
    df_test = pd.read_csv(f"{path}test_{dataset_name}.csv")
    
    tr_list = df_train['input'].tolist()
    df_train['input'] = [i.rstrip("\n") for i in tr_list]
    vl_list = df_val['input'].tolist()
    df_val['input'] = [i.rstrip("\n") for i in vl_list]
    ts_list = df_test['input'].tolist()
    df_test['input'] = [i.rstrip("\n") for i in ts_list]
    
    return df_train, df_val, df_test

def ZINC_parts(dataset_name, path):
    
    print(path)
    print(dataset_name)
    data = pd.read_csv(f"{path}/{dataset_name}.csv", sep="/t")
    import random
    n = len(data)
    l = [0]*int(n/2) + [1]*(n - int(n/2)) 
    random.shuffle(l)
    data["Classification"] = l

    df_train = data[: int(n/3)]
    df_val = data[int(n/3) : int(2*n/3)]
    df_test =data[int(2*n/3) :]

    return df_train, df_val, df_test


def USPTO_50k(dataset_name, path):
    uspto = pd.read_csv(f"{path}{dataset_name}.csv")
    smiles_reaction = uspto["reactions"]
    smiles_list = [react.split(">") for react in smiles_reaction ]
    in_react = [i[0] for i in smiles_list]
    h_react = [i[1] for i in smiles_list if i == ''] # lenght is 0
    out_react = [o[2] for o in smiles_list]
    inp_list_0 = [inp.split(".")[0] for inp in in_react ]
    inp_list_other = list(chain.from_iterable([inp.split(".")[1:] for inp in in_react ]))

    out_list = [out.split(".")[0] for out in out_react ]

    print(f"uspto lenght of first input molecule list: { len(inp_list_0)}  and 2nd and 3rd: {len(inp_list_other)}"  )
    print(f"uspto lenght of output molecule list: {len(out_list)}")
    my_list = [0]*int(len(inp_list_0)/2) + [1]*int(len(inp_list_0)/2)
    train = {"SMILES": inp_list_0, "Classification": my_list}
    test = {"SMILES": out_list, "Classification": my_list}
    my_list = [0]*(int(len(inp_list_other)/2)) + [1]*(int(len(inp_list_other)/2)+1)
    valid = {"SMILES": inp_list_other, "Classification": my_list}
    train_df = pd.DataFrame(train)
    valid_df = pd.DataFrame(valid)
    test_df = pd.DataFrame(test)
    return train_df, valid_df, test_df


def split_train_val_test(df, smiles_col_name, label_col_name):

    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    x_train, x_test, y_train, y_test = train_test_split(df['SMILES'],df['Classification'], 
                                                        test_size=1 - train_ratio)

    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    x_train = x_train.to_list()
    y_train = y_train.to_list()
    x_val = x_val.to_list()
    y_val = y_val.to_list()
    x_test = x_test.to_list()
    y_test = y_test.to_list()

    train_df = pd.DataFrame({smiles_col_name: x_train, label_col_name: y_train })    
    valid_df = pd.DataFrame({smiles_col_name: x_val, label_col_name: y_val })    
    test_df = pd.DataFrame({smiles_col_name: x_test, label_col_name: y_test })
    
    return train_df, valid_df, test_df

def cross_val(df, smiles_col_name, label_col_name, k_fold):
    # train_ratio = 0.8
    # validation_ratio = 0.1
    # test_ratio = 0.1

    # x_train, x_test, y_train, y_test = train_test_split(df['SMILES'],df['Classification'], 
    #                                                 test_size=1 - train_ratio -validation_ratio)
    # test_df = pd.DataFrame({smiles_col_name: x_test, label_col_name: y_test })
    x_train = pd.DataFrame({'ind':df['SMILES'].index, smiles_col_name: df['SMILES'].values})
    y_train = pd.DataFrame({'ind':df['Classification'].index, label_col_name: df['Classification'].values})


    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42).split(x_train, y_train)
    kf_data = list()
    for train_index, val_index in kf:
        X_train_, X_val_ = x_train.iloc[train_index], x_train.iloc[val_index]
        y_train_, y_val_ = y_train.iloc[train_index], y_train.iloc[val_index]
        train_df = pd.merge(X_train_, y_train_, on="ind")   
        valid_df = pd.merge(X_val_, y_val_, on='ind') 
        kf_data.append([train_df,valid_df])   

    return kf_data #, test_df

def read_file(ind_path):

    ind_list = []
    with open(ind_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            ind_list.append(np.array(list(map(int, line[0].split(","))))-1)
    return ind_list

def Ames_():
    dataset_name = 'Ames_fingerprint'
    store_path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data"
    path = f"{store_path}/{dataset_name}/{dataset_name}/"
    inp = "/home/gayane/BartLM"
    dataset_name = "Ames_"


    with open('/home/gayane/BartLM/fairseq/scripts/datasets.json') as f:
        datasets_json = json.load(f)

    dataset = datasets_json[dataset_name]
    si = dataset['smiles_index']
    fpi = dataset['fp_index']
    smiles_col_name, label_col_name, fp_col_name = dataset["smiles_col_name"], dataset['label_col_name'], dataset['fp_col_name']
    
    dataset_name = "Ames"
    # df = ames(dataset_name, path, smiles_col_name, label_col_name)

    df = pd.read_csv("/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/Ames_fingerprint/Ames_train.csv")

    stat_train = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/Ames/Ames/splits_train_N6512.csv"
    train_ind_list = read_file(stat_train)

    _valid = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/Ames/Ames/splits_test_N6512.csv"
    valid_ind_list = read_file(_valid)

    test_df = pd.read_csv("/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/Ames_fingerprint/Ames_external.csv")
    
    cv_train_valid_ind_list = []
    for i in range(len(valid_ind_list)):
        test_ind = valid_ind_list[i]
        train_ind = train_ind_list[i]

        dataset_i = f"{dataset_name}_{i}"
        path = f"{store_path}/{dataset_i}-fold"
        os.system(f'mkdir -p {path}')
        os.system(f'mkdir -p {path}/{dataset_i}-fold')
        train_df, valid_df = df.iloc[train_ind], df.iloc[test_ind]
        
        test_df.to_csv(f"{path}/{dataset_i}-fold/test_{dataset_name}.csv")
        train_df.to_csv(f"{path}/{dataset_i}-fold/train_{dataset_name}.csv")
        valid_df.to_csv(f"{path}/{dataset_i}-fold/valid_{dataset_name}.csv")

        loc = valid_df.columns.get_loc
        ci = dataset['class_index'][0]
        class_train = list(map(int, train_df.iloc[:, ci].tolist()))
        class_val = list(map(int, valid_df.iloc[:, ci].tolist()))
        class_test = list(map(int, test_df.iloc[:, ci].tolist()))
        smiles_train = list(map(str, train_df.iloc[:, si].tolist()))
        smiles_val = list(map(str, valid_df.iloc[:, si].tolist()))
        smiles_test = list(map(str, test_df.iloc[:, si].tolist()))
        fp_train = list(map(str, train_df.iloc[:, fpi].tolist()))
        fp_val = list(map(str, valid_df.iloc[:, fpi].tolist()))
        fp_test = list(map(str, test_df.iloc[:, fpi].tolist()))

        os.system(f"mkdir -p {path}/raw")
        os.system(f"mkdir -p {path}/tokenized")
        os.system(f"mkdir -p {path}/processed")
        os.system(f'mkdir -p {path}/processed/label')
        os.system(f'mkdir -p {path}/processed/input0')
        os.system(f'mkdir -p {path}/processed/input1')
        with open(f"{path}/processed/label/train.label", "w") as f:
            for item in class_train:
                f.write("%s\n" % item)
        with open(f"{path}/processed/label/valid.label", "w") as f:
            for item in class_val:
                f.write("%s\n" % item)
        with open(f"{path}/processed/label/test.label", "w") as f:
            for item in class_test:
                f.write("%s\n" % item)

        print(f"{dataset_i} Train Length: {len(smiles_train)}")
        print(f"{dataset_i} Valid Length: {len(smiles_val)}")
        print(f"{dataset_i} Test Length: {len(smiles_test)}")

        names = ["train", "valid", "test"]
        y_splits = []

        # Write Raw Splits
        print("Writing Input0 Splits")
        X_splits = create_raw(path, names, smiles_train, smiles_val, smiles_test, file_output = ".input0")

        print("Writing Input1 Splits")
        FP_splits = create_raw(path, names, fp_train, fp_val, fp_test, file_output = ".input1")
        
        print("Writing Output Splits")
        y_splits = create_raw(path, names, class_train, class_val, class_test, file_output = ".target")
               
        # Tokenize Texts    
        X_splits = tokenize(X_splits, inp)
        FP_splits = tokenize(FP_splits, inp)

        fairseq_preprocess_cmd(X_splits[0], X_splits[1], X_splits[2], "input0", store_path, f"{dataset_i}-fold")
        fairseq_preprocess_cmd(FP_splits[0], FP_splits[1], FP_splits[2], "input1", store_path, f"{dataset_i}-fold")
        fairseq_preprocess_cmd(y_splits[0], y_splits[1], y_splits[2], "label", store_path, f"{dataset_i}-fold")

Ames_()

def MicroNucleose():
    
    dataset_name = 'Genotoxicity'
    store_path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data"
    path = f"{store_path}/{dataset_name}"
    inp = "/home/gayane/BartLM"


    with open('/home/gayane/BartLM/fairseq/scripts/datasets.json') as f:
        datasets_json = json.load(f)

    dataset = datasets_json[dataset_name]
    si = dataset['smiles_index']
    smiles_col_name, label_col_name = dataset["smiles_col_name"], dataset['label_col_name']

    df = pd.read_csv(f"{path}/{dataset_name}/train.csv")
    k_fold = 5
    kf = KFold(n_splits=k_fold) # Define the split - into 2 folds 

    test_df = pd.read_csv(f"{path}/{dataset_name}/test.csv")
    kf_data = cross_val(df, smiles_col_name, label_col_name, k_fold)
    
    for i in range(len(kf_data)):
        train, valid = kf_data[i][0], kf_data[i][1]

        dataset_i = f"{dataset_name}_{i}"
        path = f"{store_path}/{dataset_i}-fold"
        os.system(f'mkdir -p {path}')
        os.system(f'mkdir -p {path}/{dataset_i}-fold')
        train_df, valid_df = train, valid
        
        test_df.to_csv(f"{path}/{dataset_i}-fold/test_{dataset_name}.csv")
        train_df.to_csv(f"{path}/{dataset_i}-fold/train_{dataset_name}.csv")
        valid_df.to_csv(f"{path}/{dataset_i}-fold/valid_{dataset_name}.csv")

        loc = valid_df.columns.get_loc
        ci = dataset['class_index'][0]
        class_train = list(map(int, train_df.iloc[:, ci].tolist()))
        class_val = list(map(int, valid_df.iloc[:, ci].tolist()))
        class_test = list(map(int, test_df.iloc[:, ci].tolist()))
        smiles_train = list(map(str, train_df.iloc[:, si].tolist()))
        smiles_val = list(map(str, valid_df.iloc[:, si].tolist()))
        smiles_test = list(map(str, test_df.iloc[:, si].tolist()))

        os.system(f"mkdir -p {path}/raw")
        os.system(f"mkdir -p {path}/tokenized")
        os.system(f"mkdir -p {path}/processed")
        os.system(f'mkdir -p {path}/processed/label')
        os.system(f'mkdir -p {path}/processed/input0')
        with open(f"{path}/processed/label/train.label", "w") as f:
            for item in class_train:
                f.write("%s\n" % item)
        with open(f"{path}/processed/label/valid.label", "w") as f:
            for item in class_val:
                f.write("%s\n" % item)
        with open(f"{path}/processed/label/test.label", "w") as f:
            for item in class_test:
                f.write("%s\n" % item)

        print(f"{dataset_i} Train Length: {len(smiles_train)}")
        print(f"{dataset_i} Valid Length: {len(smiles_val)}")
        print(f"{dataset_i} Test Length: {len(smiles_test)}")

        names = ["train", "valid", "test"]

        # Write Raw Splits
        print("Writing Input Splits")
        X_splits = create_raw(path, names, smiles_train, smiles_val, smiles_test, file_output = ".input")
        
        print("Writing Output Splits")
        y_splits = create_raw(path, names, class_train, class_val, class_test, file_output = ".target")
            
        # Tokenize Texts
        X_splits = tokenize(X_splits, inp)

        fairseq_preprocess_cmd(X_splits[0], X_splits[1], X_splits[2], "input0", store_path, f"{dataset_i}-fold")
        fairseq_preprocess_cmd(y_splits[0], y_splits[1], y_splits[2], "label", store_path, f"{dataset_i}-fold")