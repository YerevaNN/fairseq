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
from fairseq.data import data_utils
from fairseq import checkpoint_utils, options, quantization_utils, tasks, utils
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import networkx as nx
import torch
import numpy as np
import argparse
import logging
import math
import os
import sys
from typing import Callable, List, Optional, Tuple

from collections import OrderedDict
from matplotlib import pyplot as plt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd
from fairseq.models.bart import BARTModel

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

    datasets = cfg.model.umap_datasets.split(",")
    checkpoints = cfg.model.umap_checkpoints.split(",")

    assert len(datasets) == len(checkpoints)

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
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr, datasets, checkpoints,
                                          model=model, criterion=criterion, quantizer=quantizer)
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


# pooling: last
# pooling: take the "important' features
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


def extract_features(itr, model, pooling, num_search_batches=-1):
    subsetX_list = []
    subsetY_list = []
    with torch.no_grad():
        for i, sample in enumerate(itr):
            if num_search_batches > -1 and i >= num_search_batches:
                break
            # for sample in samples:
            subsetX_sample = move_to(sample['net_input'], device)
            subsetY_sample = move_to(sample['target'], device)

            output_tok, _ = model(**subsetX_sample)
            output = pool(pooling, output_tok, input_tok=subsetX_sample)
            subsetX_list.append(output)
            subsetY_list.append(subsetY_sample)

    subsetX = torch.concat(subsetX_list, dim=0)
    subsetY = torch.concat(subsetY_list, dim=0)

    return subsetX, subsetY


def plot(embedding, Y, dataset, log_save_dir, subset, n_neighbor, min_dist):
    data = {
        "x": embedding[:, 0].tolist(),
        "y": embedding[:, 1].tolist(),
        "label": Y.squeeze().tolist(),
        "dataset": dataset
    }
    df = pd.DataFrame(data=data)

    nodes = np.arange(df.shape[0])

    G = nx.Graph()
    G.add_nodes_from(nodes)
    # pos = [[df['x'][i], df['y'][i]] for i in range(df.shape[0])]
    pos = nx.spring_layout(G)
    all_datasets = list(set(df["dataset"]))
    edges = []
    for i in all_datasets:
        A = df[df['dataset'] == i]
        # print("--->", A.index.to_list())
        n_ = A.shape[0] # number of scaffold types
        idx = [k for k in range(n_)]
        A['idx'] = idx
        ed_p = A.index.to_list()
        for i in range(len(ed_p)):
            for j in range(i + 1, len(ed_p)):

                edges.append((ed_p[i], ed_p[j]))

    print(edges)
    for i in range(len(edges)):

        add_edge_to_graph(G, edges[i][0], edges[i][1])
    
    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, ax=ax, node_size=50, edge_color="darkgrey", alpha=0.5, width=0.5,  node_shape='o', with_labels=False)
    
    plt.axis("on")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=False)

    # sns.scatterplot(data=df, x="x", y="y", hue="dataset", style="label", alpha=0.85)
    # sns.kdeplot(data=df, x="x", y="y", shade=True, hue="dataset", alpha=0.85)
    plt.savefig(log_save_dir.joinpath(f"{subset}-{n_neighbor}-{min_dist}.png"))
    plt.clf()

def choose_n_to_k(n):
    l = list()
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            l.append((i-1, j-1))
    return l

def add_edge_to_graph(G, e1, e2):
    G.add_edge(e1, e2)

def fit_and_plot(X, Y, dataset, dir_name, subset, pooling, umap_fit_policy):
    log_save_dir = Path(f"./umap-graph-all-spring-layout/{pooling}/{umap_fit_policy}")
    log_save_dir.mkdir(parents=True, exist_ok=True)

    plt.gca().set_aspect('equal', 'datalim')
    plt.legend(fontsize=14, markerscale=2, facecolor='w')

    n_neighbors = [10, 50, 100]
    min_dists = [0.0, 0.25, 0.5, 0.99]
    for n_neighbor in n_neighbors:
        for min_dist in min_dists:
            logging.info(f"--------------------- {subset}-{n_neighbor}-{min_dist} ---------------------")
            plt.title(
                f"UMAP projection of {subset} - number of neighbours:{n_neighbor}, min distance:{min_dist}", fontsize=15)
            reducer = umap.UMAP(n_neighbors=n_neighbor, min_dist=min_dist)

            if umap_fit_policy == "grouped":
                # ground_embedding = reducer.fit_transform(groundX)
                embedding = reducer.fit_transform(X)
            elif umap_fit_policy == "seperate":
                # ground_embedding = reducer.fit_transform(groundX)
                embeddings = []
                curr_da = dataset[0]
                pivot = 0
                for i in range(len(dataset)):
                    if dataset[i] != curr_da:
                        sub_X = X[pivot:i, :]
                        embedding = reducer.fit_transform(sub_X)
                        embeddings.append(embedding)
                        curr_da = dataset[i]
                        pivot = i
                    if i == len(dataset) - 1:
                        sub_X = X[pivot:, :]
                        embedding = reducer.fit_transform(sub_X)
                        embeddings.append(embedding)
                embedding = np.concatenate(embeddings)
            else:
                raise NotImplementedError(f"umap_fit_policy method {umap_fit_policy}")

            plot(embedding, Y, dataset, log_save_dir, subset, n_neighbor, min_dist)


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr, datasets, checkpoints, **kwargs
) -> Tuple[List[Optional[float]], bool]:
    rc_dict = {
        'figure.figsize': (12.8, 9.675),
        'figure.dpi': 300,
        'figure.titlesize': 35,
        'axes.titlesize': 35,
        'axes.labelsize': 35,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.grid': False,
        'axes.axisbelow': 'line',
        'axes.labelcolor': 'black',
        'figure.facecolor': 'white',
        'grid.color': '#ffffff',
        'grid.linestyle': '-',
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        #  'lines.solid_capstyle': <CapStyle.projecting: 'projecting'>,
        'patch.edgecolor': 'black',
        'patch.force_edgecolor': False,
        'image.cmap': 'viridis',
        'font.family': ['sans-serif'],
        'font.sans-serif': ['DejaVu Sans',
                            'Bitstream Vera Sans',
                            'Computer Modern Sans Serif',
                            'Lucida Grande',
                            'Verdana',
                            'Geneva',
                            'Lucid',
                            'Arial',
                            'Helvetica',
                            'Avant Garde',
                            'sans-serif'],
        'xtick.bottom': True,
        'xtick.top': False,
        'ytick.left': True,
        'ytick.right': True,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.right': True,
        'axes.spines.top': True}

    sns.set(rc=rc_dict)
    sns.despine()

    # bart.model
    """Train the model for one epoch and return validation losses."""
    trainer.model.eval()

    # ground_dataset = cfg.model.ground_dataset
    # trivial caching
    dir_name = "".join([dataset.rstrip("/").split("/")[-2].lower() for dataset in datasets])
    cache_dir = Path("cache").joinpath("dir_name")

    if cache_dir.exists():
        trXs_stack = torch.load(cache_dir.joinpath("trXs_stack.pt"))
        trYs_stack = torch.load(cache_dir.joinpath("trYs_stack.pt"))
        vaXs_stack = torch.load(cache_dir.joinpath("vaXs_stack.pt"))
        vaYs_stack = torch.load(cache_dir.joinpath("vaYs_stack.pt"))
        valDatasets = torch.load(cache_dir.joinpath("valDatasets.pt"))
        trainDatasets = torch.load(cache_dir.joinpath("trainDatasets.pt"))

        # trainGroundX = torch.load(cache_dir.joinpath("trainGroundX.pt"))
        # trainGroundY = torch.load(cache_dir.joinpath("trainGroundY.pt"))
        # valGroundX = torch.load(cache_dir.joinpath("valGroundX.pt"))
        # valGroundY = torch.load(cache_dir.joinpath("valGroundY.pt"))
    else:
        valDatasets = []
        vaXs = []
        vaYs = []
        trainDatasets = []
        trXs = []
        trYs = []
        for dataset, checkpoint in zip(datasets, checkpoints):
            dataset_name = dataset.rstrip("/").split("/")[-2].lower()
            print(f"dataset {dataset_name}")
            cfg.model.data = dataset
            cfg.task.data = dataset
            cfg.data = dataset
            cfg.checkpoint.restore_file = checkpoint
            cfg.model.restore_file = checkpoint

            sub_task = tasks.setup_task(cfg.task)
            model = sub_task.build_model(cfg.model)
            criterion = sub_task.build_criterion(cfg.criterion)
            sub_trainer = Trainer(cfg, sub_task, kwargs["model"], kwargs["criterion"], kwargs["quantizer"])

            # val
            for valid_sub_split in cfg.dataset.valid_subset.split(","):
                sub_task.load_dataset(valid_sub_split, combine=False, epoch=1)
            itr_valid = sub_trainer.get_valid_iterator("valid").next_epoch_itr(shuffle=False, set_dataset_epoch=False)
            vaX, vaY = extract_features(itr_valid, trainer.model, cfg.model.pool)
            vaX = vaX.cpu()
            vaY = vaY.cpu()

            # if dataset_name == ground_dataset:
            #     assert valGroundX not in locals()
            #     assert valGroundY not in locals()
            #     valGroundX = vaX
            #     valGroundY = vaY
            # else:
            vaXs.append(vaX)
            vaYs.append(vaY)
            valDatasets += [dataset_name]*vaX.shape[0]

            # train
            epoch_itr = sub_trainer.get_train_iterator(
                epoch=1, load_dataset=True, disable_iterator_cache=sub_task.has_sharded_data("train")
            )
            itr_train = epoch_itr.next_epoch_itr(fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus, shuffle=(
                epoch_itr.next_epoch_idx > cfg.dataset.curriculum))
            trX, trY = extract_features(itr_train, trainer.model, cfg.model.pool)
            print("------------------------------------------")
            print(f"{checkpoint} has dim: {trX.shape}")
            print("\n")
            trX = trX.cpu()
            trY = trY.cpu()

            # if dataset_name == ground_dataset:
            #     assert trainGroundX not in locals()
            #     assert trainGroundY not in locals()
            #     trainGroundX = trX
            #     trainGroundY = trY
            # else:
            trXs.append(trX)
            trYs.append(trY)
            trainDatasets += [dataset_name]*trX.shape[0]

            del sub_trainer
            del sub_task
            del epoch_itr
            # torch.cuda.empty_cache()

        trXs_stack = torch.concat(trXs, dim=0)
        trYs_stack = torch.concat(trYs, dim=0)
        vaXs_stack = torch.concat(vaXs, dim=0)
        vaYs_stack = torch.concat(vaYs, dim=0)

        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(trXs_stack, cache_dir.joinpath("trXs_stack.pt"))
        torch.save(trYs_stack, cache_dir.joinpath("trYs_stack.pt"))
        torch.save(vaXs_stack, cache_dir.joinpath("vaXs_stack.pt"))
        torch.save(vaYs_stack, cache_dir.joinpath("vaYs_stack.pt"))
        torch.save(valDatasets, cache_dir.joinpath("valDatasets.pt"))
        torch.save(trainDatasets, cache_dir.joinpath("trainDatasets.pt"))

        # torch.save(trainGroundX, cache_dir.joinpath("trainGroundX.pt"))
        # torch.save(trainGroundY, cache_dir.joinpath("trainGroundY.pt"))
        # torch.save(valGroundX, cache_dir.joinpath("valGroundX.pt"))
        # torch.save(valGroundY, cache_dir.joinpath("valGroundY.pt"))

    sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

    for subset, X, Y, dataset in [("valid", vaXs_stack, vaYs_stack,  valDatasets), ("train", trXs_stack, trYs_stack, trainDatasets)]:
        fit_and_plot(X, Y, dataset, dir_name,
                     subset, cfg.model.pool, cfg.model.umap_fit_policy)

    sys.exit()


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

    parser.add_argument(
        "--umap-fit-policy",
        type=str,
        default="grouped",
        choices=["grouped", "seperate"],
        help="Method of fitting the umap. grouped: fits all the embs concatenated. seperate: fits each dataset emb seperatly",
    )

    parser.add_argument(
        "--umap-datasets",
        type=str,
        default="",
        help="list of datasets to plot on, comma seperate",
    )

    parser.add_argument(
        "--umap-checkpoints",
        type=str,
        default="",
        help="list of checkpoints of each dataset, comma seperate",
    )

    # parser.add_argument(
    #     "--ground-dataset",
    #     type=str,
    #     default="",
    #     help="the name of the dataset for which to plot the contoures (pre-training dataset)",
    # )

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


if __name__ == "__main__":
    cli_main()
