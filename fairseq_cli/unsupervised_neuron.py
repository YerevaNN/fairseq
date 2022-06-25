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

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PADD_IDX = 1


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


def init_reg_cv(trX, trY, vaX, vaY, penalty='l1',
                C=2**np.arange(-8, 1).astype(np.float), seed=42):
    """
    trX: [n_samples, n_units]
    trY: [n_samples, 1]
    """
    # finding the best 'c'
    # trX = trX.cpu().numpy()
    # trY = trY.cpu().numpy().ravel()
    # vaX = vaX.cpu().numpy()
    # vaY = vaY.cpu().numpy().ravel()

    # scores = []
    # for i, c in enumerate(C):
    #     model = LogisticRegression(solver="liblinear", C=c, penalty=penalty, random_state=seed+i)
    #     model.fit(trX, trY)
    #     score = model.score(vaX, vaY)
    #     scores.append(score)
    # c = C[np.argmax(scores)]

    return [LogisticRegression(solver="liblinear", C=c, penalty=penalty, random_state=seed+len(C)) for c in C]


def train_with_reg_cv(model, trX, trY, penalty='l1', seed=42):
    """
    trX: [n_samples, n_units]
    trY: [n_samples, 1]
    """
    trX = trX.cpu().numpy()
    trY = trY.cpu().numpy().ravel()

    model.fit(trX, trY)
    nnotzero = np.sum(model.coef_ != 0)
    nonzero_positions = np.nonzero(np.squeeze(model.coef_, 0))[0]

    return nnotzero, nonzero_positions


def score(model, vaX, vaY):
    vaX = vaX.cpu().numpy()
    vaY = vaY.cpu().numpy().ravel()

    return model.score(vaX, vaY)*100., roc_auc_score(vaY, model.predict_proba(vaX)[:, 1])*100


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


def pool(pooling: str, output_tok: torch.Tensor, **kwargs):
    if pooling == "avg":
        return torch.mean(output_tok, dim=1)
    elif pooling == "last":
        assert kwargs["input_tok"]
        src_tokens = kwargs["input_tok"]['src_tokens'].cpu().detach().clone().numpy()
        last_tokens = []
        for i, sample in enumerate(src_tokens):
            for j, token in enumerate(sample):
                if token == PADD_IDX:
                    last_tokens.append(j-1)
                    break
                elif j == src_tokens.shape[-1] - 1:
                    last_tokens.append(j)
        output_tokens = []
        for i, sample in enumerate(output_tok):
            output_tokens.append(sample[last_tokens[i]])
        return torch.stack(output_tokens)
    else:
        raise NotImplemented(f"pooling method '{pooling}' is not implemented")


def extract_features(itr, trainer, pooling, num_search_batches=-1):
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
            output = pool(pooling, output_tok, input_tok=subsetX_sample)
        subsetX_list.append(output)
        subsetY_list.append(subsetY_sample)

    subsetX = torch.concat(subsetX_list, dim=0)
    subsetY = torch.concat(subsetY_list, dim=0)

    return subsetX, subsetY


def plot_mlot(reg_model, nnotzero, nonzero_positions, acc, roc_auc, trX, trY, vaX, vaY, model_path, data_path, pooling):
    print("****************************************")
    print(f"   {reg_model.C} - C")
    print(f"   {nnotzero} - features used")
    print(f"   {list(nonzero_positions)} - features indices used")
    print(f"   {acc} - val accuracy")
    print(f"   {roc_auc} - val roc_auc")
    print("****************************************")

    data_path = data_path.rstrip("/").lstrip("/").split("/")[-2]
    model_path = ".".join(model_path.rstrip("/").lstrip("/").split("/")[-2:])

    log_save_dir = Path(f"./un-logs/{pooling}/{data_path}/{model_path}")
    log_save_dir.mkdir(parents=True, exist_ok=True)

    dist_save_dir = Path(f"./un-logs/{pooling}/{data_path}/{model_path}/dist")
    dist_save_dir.mkdir(parents=True, exist_ok=True)

    auc_roc_save_dir = Path(f"./un-logs/{pooling}/{data_path}/{model_path}/auc_roc")
    auc_roc_save_dir.mkdir(parents=True, exist_ok=True)

    logs = [f"{reg_model.C} - C", f"{nnotzero} - features used",
            f"{list(nonzero_positions)} - features indices used",
            f"{acc} - val accuracy", f"{roc_auc} - val roc_auc"]

    with open(log_save_dir.joinpath(f"{reg_model.C}.log.txt"), "w") as f:
        for log in logs:
            f.write(log)
            f.write("\n")

    # visualize sentiment unit
    trX_plot = trX.cpu().numpy()
    trY_plot = trY.cpu().numpy().ravel()
    sentiment_unit = trX_plot[:, 1000]
    plt.hist(sentiment_unit[trY_plot == 0], bins=25, alpha=0.5, label='neg')
    plt.hist(sentiment_unit[trY_plot == 1], bins=25, alpha=0.5, label='pos')
    plt.legend()
    plt.tight_layout()
    plt.savefig(dist_save_dir.joinpath(f"{reg_model.C}.png"))
    plt.clf()

    vaX_plot = vaX.cpu().numpy()
    vaY_plot = vaY.cpu().numpy().ravel()
    metrics_skl.plot_roc_curve(reg_model, vaX_plot, vaY_plot)
    plt.tight_layout()
    plt.savefig(auc_roc_save_dir.joinpath(f"{reg_model.C}.png"))
    plt.clf()


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    trainer.model.eval()

    # val
    itr_valid = trainer.get_valid_iterator("valid").next_epoch_itr(shuffle=False, set_dataset_epoch=False)
    vaX, vaY = extract_features(itr_valid, trainer, cfg.model.pool)

    # search
    itr_train_search = epoch_itr.next_epoch_itr(fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus, shuffle=(
        epoch_itr.next_epoch_idx > cfg.dataset.curriculum))
    num_search_batches = 3
    trX_search, trY_search = extract_features(itr_train_search, trainer, cfg.model.pool, num_search_batches)

    reg_models = init_reg_cv(trX_search, trY_search, vaX, vaY)

    # train
    itr_train = epoch_itr.next_epoch_itr(fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus, shuffle=(
        epoch_itr.next_epoch_idx > cfg.dataset.curriculum))
    trX, trY = extract_features(itr_train, trainer, cfg.model.pool)

    for reg_model in reg_models:
        nnotzero, nonzero_positions = train_with_reg_cv(reg_model, trX, trY)
        acc, roc_auc = score(reg_model, vaX, vaY)

        plot_mlot(reg_model, nnotzero, nonzero_positions, acc, roc_auc, trX,
                  trY, vaX, vaY, cfg.model.restore_file, cfg.model.data, cfg.model.pool)

    return


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    parser.add_argument(
        "--pool",
        type=str,
        default="avg",
        help="Method of representing a sentence via tokens",
    )
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
