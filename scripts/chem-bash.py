#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import csv
from tqdm import tqdm
import os

# In[17]:


with open('/home/gayane/BartLM/fairseq/scripts/YerevaNN Chemical Results - Apr25.csv') as f:
    r = csv.DictReader(f)
    lines = [row for row in r]


# In[21]:


PATH1 = "/home/gayane/BartLM/Bart/chemical/checkpoints/evaluation_data/"
PATH2 = "/processed"
BART_PATH = "/home/gayane/BartLM/Bart/chemical/checkpoints/checkpoint_last.pt"
savePath = "/mnt/good/gayane/data/chkpt/"

bs = 16

for task in tqdm(lines):
    task_name = task['']
    print(f"\nStarting {task_name}")
    TOTAL_NUM_UPDATES = int(float(task['# of steps']))
    WARMUP_UPDATES = int(0.16 * TOTAL_NUM_UPDATES)
    num_class = 2 if task['Type'] == 'Classification' else 1
    lr = task['lr']
    print(f"lr: -----> {lr}")
    subtask_count = int(task['# of subtasks'])
    if task['Type'] == 'Classification':
        for subtask in range(subtask_count):
            name = f"{task_name}_{subtask}" if subtask_count > 1 else f"{task_name}"
            codename = f"{name}_bs_{bs}_lr_{lr}_totalNum_{TOTAL_NUM_UPDATES}_warmup_{WARMUP_UPDATES}"
            directory = f"{savePath}{codename}"
            
            cmd = f"mkdir -p {directory}"
            
            os.system(cmd)
            cmd = f"""CUDA_VISIBLE_DEVICES=0 fairseq-train {PATH1}{name}{PATH2} --update-freq {bs//2} --restore-file {BART_PATH} --wandb-project Fine_Tune_{name} --batch-size 2 --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_large --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --max-target-positions 128 --max-source-positions 128 --dropout 0.2 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --relu-dropout 0.1 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr {lr} --total-num-update {TOTAL_NUM_UPDATES} --max-update {TOTAL_NUM_UPDATES} --warmup-updates {WARMUP_UPDATES} --fp16 --keep-best-checkpoints 1 --keep-last-epochs 1 --num-classes {num_class} --save-dir {directory} >> /home/gayane/BartLM/fairseq/{codename}.log"""
            print(cmd)
            os.system(cmd)
            print(f"\n   {subtask+1}/{subtask_count}: Running the following command:")
    else:
        
        codename = f"{task_name}_bs_{bs}_lr_{lr}_totalNum_{TOTAL_NUM_UPDATES}_warmup_{WARMUP_UPDATES}"
        directory = f"{savePath}{codename}"  

        cmd = f"mkdir -p {directory}"
            
        os.system(cmd) 
        cmd = f"""CUDA_VISIBLE_DEVICES=0 fairseq-train {PATH1}{task_name}{PATH2} --update-freq {bs//2} --restore-file {BART_PATH} --wandb-project Fine_Tune_{task_name} --batch-size 2 --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --init-token 0 --arch bart_large --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --max-target-positions 128 --max-source-positions 128 --dropout 0.2 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --relu-dropout 0.1 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr {lr} --total-num-update {TOTAL_NUM_UPDATES} --max-update {TOTAL_NUM_UPDATES} --warmup-updates {WARMUP_UPDATES} --keep-last-epochs 1 --keep-best-checkpoints 1 --fp16 --threshold-loss-scale 1 --fp16-scale-window 128 --max-epoch 10 --best-checkpoint-metric loss --regression-target --num-classes {num_class} --save-dir {directory} >> {directory}/{codename}.log"""
        print(f"\n   {1}/{subtask_count}: Running the following command:")
        print(cmd)
        os.system(cmd)
    print("\n\n")


print(f"\n   {subtask+1}/{subtask_count}: Running the following command:")
    



