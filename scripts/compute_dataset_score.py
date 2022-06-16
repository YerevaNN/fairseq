#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import numpy as np
import csv
import os

from os import listdir
from os.path import isfile, join



with open('/home/gayane/BartLM/fairseq/scripts/YerevaNN Chemical Results - Apr25.csv') as f:
    r = csv.DictReader(f)
    lines = [row for row in r]
    print(f"------------> {lines}")

for task in tqdm(lines):
    task_name = task['']
    subtask_count = int(task['# of subtasks'])
    lr = str(float(task['lr'])).replace('0', '')
    # if task_name != "Tox21":
    #     continue
    print(type(task_name), type(subtask_count), type(lr))
    TOTAL_NUM_UPDATES = int(float(task['# of steps']))
    WARMUP_UPDATES = int(0.16 * TOTAL_NUM_UPDATES)
    drout = task["dropout"]
    skip_set = set([i for i in range(21)])

    for subtask in range(subtask_count):
        print(f"\nStarting subtask {int(subtask)}")
        if "Ames" in task_name or task_name in set(["esol", "freesolv", "lipo", "Ames", "BBBP", "BACE", "HIV"]): 

            task_name_ = task_name 
            
        else:
            f"{task_name}_{subtask}"


        file_path = f"/mnt/good/gayane/data/chkpt/{task_name_}_bs_16_dropout_{drout}_lr_{lr}_totalNum_{TOTAL_NUM_UPDATES}_warmup_{WARMUP_UPDATES}/"
        onlyfiles = [f for f in os.listdir(file_path)]
        for file_name in onlyfiles:
            if "checkpoint_best" in file_name:
                print('skip')
                continue
            cmd = f"""python /home/gayane/BartLM/fairseq/scripts/compute_auc.py --lr {lr} --dropout {drout} --dataset-name {task_name_} --subtask {str(subtask)} --warmup-update {WARMUP_UPDATES} --total-number-update {TOTAL_NUM_UPDATES} --checkpoint_name {file_name}""" 
            print(cmd)
            os.system(cmd)
    print("\n\n")
    