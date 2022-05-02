from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


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
    df.to_csv(path +"AcuteOralToxicity.csv")

    return df

def ames(dataset_name, path):
    print (dataset_name)
    df = pd.read_csv(path + "Ames.csv")
    print(df.columns)
    print(df.head(2))
    df = df.dropna()
    return df
 
def split_train_val_test(df, smiles_col_name, label_col_name):

    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    x_train, x_test, y_train, y_test = train_test_split(df[smiles_col_name],
                                                        label_col_name = df[label_col_name], 
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


