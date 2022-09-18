from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from itertools import chain
from rdkit import Chem
import pandas as pd
import numpy as np
import rdkit
import json
import csv
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'


def generateMurcoScaffold(dataset_name):

    path = f"/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/{dataset_name}/{dataset_name}/"
    train_df = pd.read_csv(f"{path}train_{dataset_name}.csv")
    valid_df = pd.read_csv(f"{path}valid_{dataset_name}.csv")
    test_df = pd.read_csv(f"{path}test_{dataset_name}.csv")

    train_scaff = []
    valid_scaff = []
    train_scaff = []

    include_chirality = False

    for i in range(len(train_df)):
            mol = Chem.MolFromSmiles(train_df['ids'][i])
            train_scaff.append(MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality))
    train_df["MurckoScaffold"] = train_scaff
    train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]



    for i in range(len(valid_df)):
        mol = Chem.MolFromSmiles(valid_df['ids'][i])
        valid_scaff.append(MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality))
    valid_df["MurckoScaffold"] = valid_scaff
    valid_df = valid_df.loc[:, ~valid_df.columns.str.contains('^Unnamed')]

    print(train_df)
    #     print(valid_df)
    return train_df, valid_df

def scaff(dataset_name, type):

    # print (dataset_name)
    path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/"
    # file=open(path + f"{dataset_name}/{dataset_name}/{type}_{dataset_name}.csv")
    # print (dataset_name)

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

def regAcuteOralToxicity(dataset_name, path):

    print (dataset_name)
    file=open(path + "AcuteOralToxicity.txt")
    file = file.read().split('\n')
    sm = list()
    col = list()
    for i in range(1,len(file)):
        sm.append(file[i].split('\t')[4].strip('"'))
        col.append(file[i].split('\t')[-3])

    d = {"Canonical_QSARr": sm, "LD50_mgkg": col }

    df = pd.DataFrame(d) 
    df['LD50_mgkg'] = df['LD50_mgkg'].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(inplace=True)
    df.to_csv(path +"AcuteOralToxicity.csv")
    return df

def classAcuteOralToxicity(dataset_name, path):

    print (dataset_name)
    file=open(path + "AcuteOralToxicity.txt")
    file = file.read().split('\n')
    sm = list()
    col = list()
    for i in range(1,len(file)):
        sm.append(file[i].split('\t')[4].strip('"'))
        col.append(file[i].split('\t')[-2])

    d = {"Canonical_QSARr": sm, "EPA_category": col }

    df = pd.DataFrame(d) 
    # df['EPA_category'] = (df['EPA_category'] !='NA').astype(int)
    df['EPA_category'] = df['EPA_category'].apply (pd.to_numeric, errors='coerce')
    df = df.dropna()

    df.to_csv(path +"AcuteOralToxicity.csv")
    return df

def genotoxicity(dataset_name, path):
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



def japan(dataset_name, path):
    print (dataset_name)

    file=open(path + "japan.csv")
    file = file.read().split('\n')
    sm = list()
    col = list()

    for i in range(1,len(file)-1):
        sm.append(file[i].split(',')[0])
        col.append(file[i].split(',')[1].strip("'"))
    bool_col = [int(i) for i in col]

    d = {"SMILES": sm, "Classification": bool_col }
    df = pd.DataFrame(d) 
    df = df.dropna()
    df.to_csv(path +"japan_.csv")

    return df

def snyder_negatives_451(dataset_name, path):
    print (dataset_name)

    file=open(path + "snyder_negatives_451_.csv")
    file = file.read().split('\n')
    sm = list()
    col = list()

    for i in range(1,len(file)-1):
        sm.append(file[i].split(',')[2])
        col.append(float(file[i].split(',')[3].strip("'")))
    bool_col = [int(i) for i in col]

    d = {"SMILES": sm, "Classification": bool_col }
    df = pd.DataFrame(d) 
    df = df.dropna()
    df.to_csv(path +"snyder_negatives_451.csv")

    return df


def ames(dataset_name, path, smiles_col_name, label_col_name):
    print(path)
    print (f"{path}{dataset_name}.csv")
    df = pd.read_csv(f"{path}{dataset_name}.csv")
    d = {"SMILES": df[smiles_col_name], "Classification": df[label_col_name] }
    df = pd.DataFrame(d) 
    # df = df.dropna()
    print(df.columns, df.shape)
    print(df['SMILES'].head())

    
    # df = df.loc[ind_list]
    return df

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

# a = snyder_negatives_451("snyder_negatives_451", "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/snyder_negatives_451/snyder_negatives_451/")

def Ames_10_times():

    dataset_name = 'Ames'
    store_path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data"
    path = f"{store_path}/{dataset_name}_main/{dataset_name}_main/"
    inp = "/home/gayane/BartLM"

    with open('/home/gayane/BartLM/fairseq/scripts/datasets.json') as f:
        datasets_json = json.load(f)

    dataset = datasets_json[dataset_name]
    si = dataset['smiles_index']
    smiles_col_name, label_col_name = dataset["smiles_col_name"], dataset['label_col_name']


    df_cv = ames("Ames_train", path, smiles_col_name, label_col_name)
    df_ext = ames("Ames_external", path, smiles_col_name, label_col_name)

def read_file(ind_path):

    ind_list = []
    with open(ind_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            ind_list.append(np.array(list(map(int, line[0].split(","))))-1)
    return ind_list


def Ames():
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

    df = pd.read_csv(f"{store_path}/Ames_fingerprint/Ames_train.csv")

    stat_train = f"{store_path}/Ames/Ames/splits_train_N6512.csv"
    train_ind_list = read_file(stat_train)

    _valid = f"{store_path}/Ames/Ames/splits_test_N6512.csv"
    valid_ind_list = read_file(_valid)

    test_df = pd.read_csv(f"{store_path}/Ames_fingerprint/Ames_external.csv")


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
        X_splits = []
        FP_splits = []
        y_splits = []


        # Write Raw Splits
        print("Writing Input0 Splits")
        
        for name, smiles in zip(names, (smiles_train, smiles_val, smiles_test)):
            print(name + ".target")
            print(dataset_i )
            new_path = f"{path}/raw/" + name + ".input0" 
            X_splits.append(new_path)
            with open(new_path, "w+") as f:
                for smile in smiles:
                    f.write(f"{smile}\n")

        print("Writing Input1 Splits")
        
        for name, fps in zip(names, (fp_train, fp_val, fp_test)):
            print(name + ".target")
            print(dataset_i )
            new_path = f"{path}/raw/" + name + ".input1" 
            FP_splits.append(new_path)
            with open(new_path, "w+") as f:
                for fp in fps:
                    f.write(f"{fp}\n")
        
        print("Writing Output Splits")
        
        for name, targets in zip(names, (class_train, class_val, class_test)):
            print(dataset_i)
            new_path = f"{path}/raw/" + name + ".target"
            print(new_path)
            y_splits.append(new_path)
            with open(new_path, "w+") as f:
                for target in targets:
                    f.write(f"{str(target)}\n")

            
        # Tokenize Texts
        print("Tokenizing")
        splits = []
        for path_ in X_splits:
            cur_path = path_.replace('raw', 'tokenized')
            print(path_)
            print(cur_path)
            splits.append(cur_path)
            cmd = f"python {inp}/fairseq/scripts/spm_parallel.py --input {path_} --outputs {cur_path} --model /home/gayane/BartLM/Bart/chemical/tokenizer/chem.model"
            print(cmd)
            os.system(cmd)

        X_splits = splits

        splits = []
        for path_ in FP_splits:
            cur_path = path_.replace('raw', 'tokenized')
            print(path_)
            print(cur_path)
            splits.append(cur_path)
            cmd = f"python {inp}/fairseq/scripts/spm_parallel.py --input {path_} --outputs {cur_path} --model /home/gayane/BartLM/Bart/chemical/tokenizer/chem.model"
            print(cmd)
            os.system(cmd)

        FP_splits = splits



        os.system(('fairseq-preprocess --only-source '
                f'--trainpref "{FP_splits[0]}" '
                f'--validpref "{FP_splits[1]}" '
                f'--testpref "{FP_splits[2]}" '
                f'--destdir "{path}/processed/input1" --workers 60 '
                '--srcdict /home/gayane/BartLM/Bart/chemical/tokenizer/chem.vocab.fs'))

        os.system(('fairseq-preprocess --only-source '
                f'--trainpref "{X_splits[0]}" '
                f'--validpref "{X_splits[1]}" '
                f'--testpref "{X_splits[2]}" '
                f'--destdir "{path}/processed/input0" --workers 60 '
                '--srcdict /home/gayane/BartLM/Bart/chemical/tokenizer/chem.vocab.fs'))
        os.system(('fairseq-preprocess '
                '--only-source '
                f'--trainpref "{y_splits[0]}" '
                f'--validpref "{y_splits[1]}" '
                f'--testpref "{y_splits[2]}" '
                f'--destdir "{path}/processed/label" --workers 60 '))


Ames()



def mNInVitro(dataset_name, path, smiles_col_name, label_col_name):
    print(path)
    print (f"{path}{dataset_name}.csv")
    df = pd.read_csv(f"{path}/{dataset_name}.csv")
    d = {"SMILES": df[smiles_col_name], "Classification": df[label_col_name] }
    df = pd.DataFrame(d) 
    # df = df.dropna()
    print(df.columns, df.shape)
    print(df['SMILES'].head())

    
    # df = df.loc[ind_list]
    return df

def MNInVitro():
    
    dataset_name = 'MNInVitro'
    store_path = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data"
    path = f"{store_path}/{dataset_name}"
    inp = "/home/gayane/BartLM"


    with open('/home/gayane/BartLM/fairseq/scripts/datasets.json') as f:
        datasets_json = json.load(f)

    dataset = datasets_json[dataset_name]
    si = dataset['smiles_index']
    smiles_col_name, label_col_name = dataset["smiles_col_name"], dataset['label_col_name']

    df = mNInVitro(dataset_name, path, smiles_col_name, label_col_name)
    train_df, valid_df = train_test_split(df, test_size = 0.1, random_state=1)

    test_df = mNInVitro(f"{dataset_name}_main_test", path, smiles_col_name, label_col_name)
    cv_train_valid_ind_list = []
    # for i in range(len(valid_ind_list)):

    # dataset_i = f"{dataset_name}_{i}"
    # path = f"{store_path}/{dataset_i}-fold"
    os.system(f'mkdir -p {path}')
    os.system(f'mkdir -p {path}/{dataset_name}')
    # train_df, valid_df = df.iloc[train_ind], df.iloc[test_ind]
    
    test_df.to_csv(f"{path}/{dataset_name}/test_{dataset_name}.csv")
    train_df.to_csv(f"{path}/{dataset_name}/train_{dataset_name}.csv")
    valid_df.to_csv(f"{path}/{dataset_name}/valid_{dataset_name}.csv")

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

    print(f"{dataset_name} Train Length: {len(smiles_train)}")
    print(f"{dataset_name} Valid Length: {len(smiles_val)}")
    print(f"{dataset_name} Test Length: {len(smiles_test)}")

    names = ["train", "valid", "test"]
    X_splits = []
    y_splits = []


    # Write Raw Splits
    print("Writing Input Splits")
    
    for name, smiles in zip(names, (smiles_train, smiles_val, smiles_test)):
        print(name + ".target")
        print(dataset_name )
        new_path = f"{path}/raw/" + name + ".input" 
        X_splits.append(new_path)
        with open(new_path, "w+") as f:
            for smile in smiles:
                f.write(f"{smile}\n")
    print("Writing Output Splits")
    
    for name, targets in zip(names, (class_train, class_val, class_test)):
        print(dataset_name)
        new_path = f"{path}/raw/" + name + ".target"
        print(new_path)
        y_splits.append(new_path)
        with open(new_path, "w+") as f:
            for target in targets:
                f.write(f"{str(target)}\n")

        
    # Tokenize Texts
    print("Tokenizing")
    splits = []
    for path_ in X_splits:
        cur_path = path_.replace('raw', 'tokenized')
        print(path_)
        print(cur_path)
        splits.append(cur_path)
        cmd = f"python {inp}/fairseq/scripts/spm_parallel.py --input {path_} --outputs {cur_path} --model /home/gayane/BartLM/Bart/chemical/tokenizer/chem.model"
        print(cmd)
        os.system(cmd)

    X_splits = splits

    os.system(('fairseq-preprocess --only-source '
            f'--trainpref "{X_splits[0]}" '
            f'--validpref "{X_splits[1]}" '
            f'--testpref "{X_splits[2]}" '
            f'--destdir "{path}/processed/input0" --workers 60 '
            '--srcdict /home/gayane/BartLM/Bart/chemical/tokenizer/chem.vocab.fs'))
    os.system(('fairseq-preprocess '
            '--only-source '
            f'--trainpref "{y_splits[0]}" '
            f'--validpref "{y_splits[1]}" '
            f'--testpref "{y_splits[2]}" '
            f'--destdir "{path}/processed/label" --workers 60 '))


# MNInVitro()


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
        X_splits = []
        y_splits = []


        # Write Raw Splits
        print("Writing Input Splits")
        
        for name, smiles in zip(names, (smiles_train, smiles_val, smiles_test)):
            print(name + ".target")
            print(dataset_i )
            new_path = f"{path}/raw/" + name + ".input" 
            X_splits.append(new_path)
            with open(new_path, "w+") as f:
                for smile in smiles:
                    f.write(f"{smile}\n")
        print("Writing Output Splits")
        
        for name, targets in zip(names, (class_train, class_val, class_test)):
            print(dataset_i)
            new_path = f"{path}/raw/" + name + ".target"
            print(new_path)
            y_splits.append(new_path)
            with open(new_path, "w+") as f:
                for target in targets:
                    f.write(f"{str(target)}\n")

            
        # Tokenize Texts
        print("Tokenizing")
        splits = []
        for path_ in X_splits:
            cur_path = path_.replace('raw', 'tokenized')
            print(path_)
            print(cur_path)
            splits.append(cur_path)
            cmd = f"python {inp}/fairseq/scripts/spm_parallel.py --input {path_} --outputs {cur_path} --model /home/gayane/BartLM/Bart/chemical/tokenizer/chem.model"
            print(cmd)
            os.system(cmd)

        X_splits = splits

        os.system(('fairseq-preprocess --only-source '
                f'--trainpref "{X_splits[0]}" '
                f'--validpref "{X_splits[1]}" '
                f'--testpref "{X_splits[2]}" '
                f'--destdir "{path}/processed/input0" --workers 60 '
                '--srcdict /home/gayane/BartLM/Bart/chemical/tokenizer/chem.vocab.fs'))
        os.system(('fairseq-preprocess '
                '--only-source '
                f'--trainpref "{y_splits[0]}" '
                f'--validpref "{y_splits[1]}" '
                f'--testpref "{y_splits[2]}" '
                f'--destdir "{path}/processed/label" --workers 60 '))


# MicroNucleose()