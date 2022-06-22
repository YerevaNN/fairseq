#!/usr/bin/env python

import sweep
import sweep as sweep
from sweep import hyperparam

# /data/home/armenag/code/chem/evaluation_data/processed


def get_grid(args):
    grid = []

    total_num_udpates = 2296
    warmup_updates = int(total_num_udpates*0.16)
    num_data_loaders = 1
    arch = "bart_large"
    criterion = "sentence_prediction"

    adam_eps = 1e-08
    weight_decay = 0.01

    update_freq = 1
    grid += [
        hyperparam(
            "--restore-file",
            "/data/home/armenag/code/chem/models/v3/checkpoint_last.pt",
        )
    ]

    # model settings
    grid += [
        hyperparam("--arch", arch, save_dir_key=lambda val: val),
        hyperparam(
            "--task", "sentence_prediction", save_dir_key=lambda _: "sentpred"
        ),
        hyperparam("--criterion", criterion),
        hyperparam("--num-classes", 2),
        hyperparam("--max-target-positions", 512),
        hyperparam("--max-source-positions", 512),
        hyperparam("--add-prev-output-tokens", True, binary_flag=True),
    ]

    grid += [
        hyperparam("--batch-size", 16, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam(
            "--max-update", total_num_udpates, save_dir_key=lambda val: f"mu{val}"
        ),
        hyperparam("--required-batch-size-multiple", 1),
    ]
    # regularization
    grid += [
        hyperparam("--dropout", 0.2, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.2, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--relu-dropout", 0.1, save_dir_key=lambda val: f"actdr{val}"),
        hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
    ]

    # optimization settings
    grid += [
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.999)", save_dir_key=lambda val: "beta9999"),
        hyperparam("--adam-eps", adam_eps, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
    ]

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", 3e-05, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", total_num_udpates),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"warm{val}"
        ),
    ]
    grid += [
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
    ]

    # data loading settings
    grid += [
        hyperparam("--num-workers", num_data_loaders),
    ]

    # validation and checkpoint settings
    grid += [
        # hyperparam("--no-save"),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--reset-meters"),
        hyperparam("--reset-optimizer"),
        hyperparam("--reset-dataloader"),
        # hyperparam("--best-checkpoint-metric", "accuracy"),
        # hyperparam("--maximize-best-checkpoint-metric", [True], binary_flag=True),
    ]

    grid += [
        hyperparam("--share-all-embeddings"),
        hyperparam("--layernorm-embedding"),
        hyperparam("--share-decoder-input-output-embed"),
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 1),
    ]

    if args.local:
        grid += [
            hyperparam("--log-format", "json"),
            hyperparam("--log-interval", 1),
        ]
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)