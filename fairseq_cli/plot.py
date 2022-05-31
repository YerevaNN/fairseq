#!/usr/bin/env python3 -u
import os
from fairseq.trainer import Trainer
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.logging import meters, metrics
from fairseq.file_io import PathManager
from fairseq.distributed import utils as distributed_utils
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.dataclass.initialize import add_defaults
from fairseq.dataclass.configs import FairseqConfig
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.data import data_utils, iterators
from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import argparse
import logging
import math
import os
import sys
from typing import Callable, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from collections import OrderedDict
from matplotlib import pyplot as plt
from sklearn import metrics as metrics_skl
from sklearn.metrics import roc_auc_score
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)
    add_defaults(cfg)

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(
                p.numel() for p in model.parameters() if not getattr(p, "expert", False)
            ),
            sum(
                p.numel()
                for p in model.parameters()
                if not getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(
                p.numel()
                for p in model.parameters()
                if getattr(p, "expert", False) and p.requires_grad
            ),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm

        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, OrderedDict):
        res = OrderedDict()
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        return obj


def extract_features(itr, trainer, num_search_batches=-1):
    subsetX_list = []
    subsetY_list = []
    for i, sample in enumerate(itr):
        if num_search_batches > -1 and i >= num_search_batches:
            break
        # for sample in samples:
        subsetX_sample = move_to(sample['net_input'], device)
        subsetY_sample = move_to(sample['target'], device)

        with torch.no_grad():
            output_tok, _ = trainer.model(**subsetX_sample)
            output = torch.mean(output_tok, dim=1)
        subsetX_list.append(output)
        subsetY_list.append(subsetY_sample)

    subsetX = torch.concat(subsetX_list, dim=0)
    subsetY = torch.concat(subsetY_list, dim=0)

    return subsetX, subsetY


def plot_mlot(X, Y, i, subset, model_path, data_path):
    data_path = data_path.rstrip("/").lstrip("/").split("/")[-2]
    model_path = ".".join(model_path.rstrip("/").lstrip("/").split("/")[-2:])

    log_save_dir = Path(f"./umap-logs/{data_path}/{model_path}")
    log_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"working on: {subset}.{i}.png")

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(X)

    plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[y] for y in Y])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f"UMAP projection of {subset}", fontsize=24)
    plt.savefig(log_save_dir.joinpath(f"{subset}.{i}.png"))
    plt.clf()


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    trainer.model.eval()

    # val
    itr_valid = trainer.get_valid_iterator("valid").next_epoch_itr(shuffle=False, set_dataset_epoch=False)
    vaX, vaY = extract_features(itr_valid, trainer)
    vaX = vaX.cpu()
    vaY = vaY.cpu()

    # train
    itr_train = epoch_itr.next_epoch_itr(fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus, shuffle=(
        epoch_itr.next_epoch_idx > cfg.dataset.curriculum))
    trX, trY = extract_features(itr_train, trainer)
    trX = trX.cpu()
    trY = trY.cpu()

    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

    for i in range(10):
        for subset, X, Y in [("valid", vaX, vaY), ("train", trX, trY)]:
            plot_mlot(X, Y, i, subset, cfg.model.restore_file, cfg.model.data)

    return


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
