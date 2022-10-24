To fine-tune on MoleculeNet tasks

1) Download and preprocess MoleculNet datasets examples/scripts/preprocess_bartsmiles.py
This will create Folders: 

2) Generate the grid of training hyperparameters scripts/generate_grid_bartsmiles. This will create a csv.

3) Login to your wandb

4) Train the models scripts/train_grid_bartsmiles. This will produce checkpoint ...

5) Perform SWA and evaluate scripts/evaluate_  _bartsmiles. This will produce a CSV with output and averaged checkpoints in ... .
