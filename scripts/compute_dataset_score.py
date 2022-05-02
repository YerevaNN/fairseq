from tqdm import tqdm
import numpy as np
import csv
import os



with open('/home/hrant/YerevaNN Chemical Results - Apr25.csv') as f:
    r = csv.DictReader(f)
    lines = [row for row in r]


for task in tqdm(lines):
    task_name = task['']
    subtask_count = int(task['# of subtasks'])
    TOTAL_NUM_UPDATES = int(float(task['# of steps']))
    WARMUP_UPDATES = int(0.16 * TOTAL_NUM_UPDATES)
    for subtask in range(subtask_count):

        print(f"\nStarting subtask {int(subtask)}")

        cmd = f"python /home/gayane/BartLM/fairseq/scripts/compute_auc.py --dataset-name {task_name} --subtask {str(int(subtask))} --warmup-update  {WARMUP_UPDATES} --total-number-update {TOTAL_NUM_UPDATES} " 
        print(cmd)
        os.system(cmd)
