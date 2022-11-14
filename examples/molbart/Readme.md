To fine-tune on MoleculeNet tasks


1) Download and preprocess MoleculeNet datasets: `python process_datasets.py --dataset-name esol --is-MoleculeNet True` from the `root/fairseq/examples/molbart`
    This will create folders in `root/BARTSmiles/checkpoints/evaluation_data` directory: 

```
    esol
    │
    ├───esol
    │      train_esol.csv
    │      valid_esol.csv
    │      test_esol.csv
    │
    │
    ├───processed
    │   │
    │   ├───input0
    │   │       dict.txt
    │   │       preprocess.log
    │   │       test.bin
    │   │       train.bin
    │   │       valid.bin
    │   │       test.idx
    │   │       valid.idx
    │   │       train.idx
    │   │
    │   └───label
    │          dict.txt
    │          preprocess.log
    │          test.bin
    │          valid.bin
    │          train.bin
    │          test.idx
    │          valid.idx
    │          train.idx 
    │          test.label
    │          valid.label
    │          train.label
    │
    │
    ├───raw
    |      test.input
    |      test.target
    |      valid.input
    |      valid.target
    |      train.input
    |      train.target
    |   
    |
    |
    └───tokenized
        test.input
        valid.input
        train.input
```


2) Generate the grid of training hyperparameters /home/gayane/BartLM/fairseq/examples/molbart/generate_grid_bartsmiles.py. This will create a csv.
    cmd for regression task: python generate_grid_bartsmilels.py --dataset-name esol --is-Regression True 
    cmd for classification single task: python generate_grid_bartsmilels.py --dataset-name BBBP --single-task True
    cmd for classification multilabel task: python generate_grid_bartsmilels.py --dataset-name tox21 --single-task False --subtask 12

    This will write grid search parameters in "/home/gayane/BartLM/fairseq/examples/molbart/YerevaNN Chemical Results - Apr25.csv" file.

3) Login to your wandb
    You have to login in wandb.
    You can follow: https://docs.wandb.ai/ref/cli/wandb-login 


4) Train the models 
CMD: python /home/gayane/BartLM/fairseq/examples/molbart/train_grid_bartsmiles.py. 
This will produce checkpoint in /mnt/good/gayane/data/chkpt/clintox_1_bs_16_dropout_0.1_lr_5e-6_totalNum_739_warmup_118/ folder.

5) You will write wandb url in wandb_url.csv file 
example:
url
gayanec/Fine_Tune_clintox_0/6p76cyzr

6) Perform SWA and evaluate /home/gayane/BartLM/fairseq/examples/molbart/evaluate_swa_bartsmiles.py. This will produce a log file with output and averaged checkpoints respectivly in   /home/gayane/BartLM/Bart/chemical/log/  and /mnt/good/gayane/data/chkpt/clintox_1_bs_16_dropout_0.1_lr_5e-6_totalNum_739_warmup_118/ folders.
