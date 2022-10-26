import pandas as pd
import argparse


p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset-name", 
                    type=str, 
                    required=True)
p.add_argument("--dataset-size", 
                    type=str, 
                    default="4200")
p.add_argument("--subtasks", 
                    type=int, 
                    default=0)
p.add_argument("--single-task", 
                    type=bool, 
                    default=True)
p.add_argument("--epoch", 
                    type=int, 
                    default=10)
p.add_argument("--batch-size", 
                    type=int, default=16)

p.add_argument("--is-Regression", 
                    default=False, 
                    help="Regrestion։ True or Classification: False")

p.add_argument("--add-noise", 
                    type=bool,
                    help="True or False")

args = p.parse_args()
name = args.dataset_name
ep = args.epoch
bs = args.batch_size
dataset_size = int(args.dataset_size)
subtask = args.subtasks
subtask = 0 if args.single_task else subtask
TOTAL_NUM_UPDATES = (dataset_size * 0.8) / (ep * bs)

time_param = 5.2 / 7000

head = ["","Type","Experimental","Datasize","# of steps","# of subtasks","lr","Minutes to train 1 subtask","Hours to train all subtasks","dropout","noise_type","lambda"]

regr_or_class = "Regression" if args.is_Regression else "Classification"
minuts_1_task = dataset_size * 0.8 * time_param * 10
Hours_to_train_all_subtasks = minuts_1_task * (subtask + 1) / 60 if args.single_task else minuts_1_task * time_param * subtask / 6

minuts_1_task = round(minuts_1_task, 1)
Hours_to_train_all_subtasks = round(Hours_to_train_all_subtasks, 1)
# Genotoxicity_3-fold,Classification,Yes,448,290,1,3e-5,25,0.4,0.1
learning_rate = ["5e-6", "1e-5", "3e-5"]
dropouts = [ 0.1, 0.2, 0.3 ] 

lmb = [0.1, 0.5, 1.0, 5.0]
noise = ["uniform", "normal"]
path = "/home/gayane/BartLM/fairseq/scripts/YerevaNN Chemical Results - Apr25.csv"
with open(path, 'w') as f:
    head = ["","Type","Experimental","Datasize","# of steps","# of subtasks","lr","Minutes to train 1 subtask","Hours to train all subtasks","dropout","noise_type","lambda"]
    f.write(",".join(head))
    for lr in learning_rate:
        for dropout in dropouts:
            if args.add_noise:
                for nt in noise:
                    for ld in lmb: 
                        row = f"{name},{regr_or_class},Yes,{dataset_size},{TOTAL_NUM_UPDATES},{(subtask + 1) if args.single_task else minuts_1_task * subtask},{lr},{minuts_1_task},{Hours_to_train_all_subtasks},{dropout},{nt},{ld}"
                        f.write(row)
                        f.write('\n')
                        
            else:
                row = f"{name},{regr_or_class},Yes,{dataset_size},{TOTAL_NUM_UPDATES},{(subtask + 1) if args.single_task else minuts_1_task * subtask},{lr},{minuts_1_task},{Hours_to_train_all_subtasks},{dropout}"
                f.write(row)
                f.write('\n')