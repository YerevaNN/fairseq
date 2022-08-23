#!/usr/bin/env python3 -u
import os
from fairseq import checkpoint_utils
from fairseq.checkpoint_utils import checkpoint_paths
from fairseq.trainer import Trainer
from fairseq.distributed import utils as distributed_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import options, tasks
from fairseq.models.transformer_lm import TransformerLanguageModel
from omegaconf import DictConfig, OmegaConf
import torch
import argparse
import logging
import sys
from typing import TypeVar, Type, Callable, List, Optional, Tuple

from collections import OrderedDict
from pathlib import Path

from sklearn.manifold import MDS
import umap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import pickle

C = TypeVar("C")

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
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})


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


class Cachable:
    cache_dir: Path

    @classmethod
    def cache(cls: Type[C]):
        def initialized(callback: Callable):
            def called(*args, **kwargs):
                cache_file_path = Cachable.cache_dir.joinpath(args[2], f"{args[3]}.pt")
                if cache_file_path.is_file():
                    with open(cache_file_path, "rb") as f:
                        callback_result = pickle.load(f)
                else:
                    callback_result = callback(*args, **kwargs)
                    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
                    with open(cache_file_path, "wb") as f:
                        pickle.dump(callback_result, f, protocol=pickle.HIGHEST_PROTOCOL)
                return callback_result
            return called
        return initialized


class Worker(Cachable):
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.subsets = ["valid", "train"]
        self.datasets_paths = self.cfg.model.umap_datasets.split(",")
        self.checkpoints_paths = self.cfg.model.umap_checkpoints.split(",")
        self.ground_dataset_name = self.cfg.model.ground_dataset
        self.max_n_batches = self.cfg.model.n_batches
        self.method = self.cfg.model.method
        self.dataset_instances = {}
        for dataset_path, checkpoint_path in zip(self.datasets_paths, self.checkpoints_paths):
            self.dataset_instances[self._dataset_name(dataset_path)] = {
                "dataset_path": dataset_path, "checkpoint_path": checkpoint_path}
            self.dataset_instances[self._dataset_name(dataset_path)].update(
                {subset: {"features": torch.TensorType, "labels": torch.TensorType} for subset in self.subsets})

        assert list(self.dataset_instances.keys()
                    )[-1] == self.ground_dataset_name, "the ground_dataset is assumed to be the last"

        self.pooling_method = cfg.model.pooling_method

        self.dir_name = "-".join([dataset_name for dataset_name in self.dataset_instances.keys()])
        self.log_dir = Path(self.cfg.model.log_dir).joinpath(self.method, self.dir_name, str(
            self.max_n_batches) if self.max_n_batches is not None else "", self.cfg.model.plot_policy, self.pooling_method, self.cfg.model.fit_policy)
        os.makedirs(self.log_dir, exist_ok=True)

        self.cache_dir = Path("cache").joinpath(self.dir_name, str(self.max_n_batches)
                                                if self.max_n_batches is not None else "", self.cfg.model.plot_policy, self.pooling_method)
        os.makedirs(self.cache_dir, exist_ok=True)

        Cachable.cache_dir = self.cache_dir

        if self.cfg.model.plot_policy == "pretrained":
            self.build_stack(self.dataset_instances[self.ground_dataset_name]["dataset_path"],
                             self.dataset_instances[self.ground_dataset_name]["checkpoint_path"])

        self.methods = {
            "umap": self.fit_transform_umap,
            "mds": self.fit_transform_mds
        }

        self.umap_transforms = {
            "separate": self.fit_transform_umap_separate,
            "grouped": self.fit_transform_umap_grouped
        }

        self.mds_transforms = {
            "seperate": self.fit_transform_mds_seperate,
            "grouped": self.fit_transform_mds_grouped,
        }

    def build_stack(self, dataset_path: str, checkpoint_path: str):
        if hasattr(self, "trainer") and self.cfg.model.plot_policy == "pretrained":
            self.cfg.model.data = dataset_path
            self.cfg.task.data = dataset_path
        else:
            self.cfg.model.data = dataset_path
            self.cfg.task.data = dataset_path
            self.cfg.checkpoint.restore_file = checkpoint_path
            self.cfg.model.restore_file = checkpoint_path

        self.task = tasks.setup_task(self.cfg.task)
        self.model = self.task.build_model(self.cfg.model)
        self.model.eval()

        self.criterion = self.task.build_criterion(self.cfg.criterion)
        self.trainer = Trainer(self.cfg, self.task, self.model, self.criterion)

        checkpoint_utils.load_checkpoint(
            self.cfg.checkpoint,
            self.trainer,
            disable_iterator_cache=self.task.has_sharded_data("train"),
        )

    def get_iterator(self, subset: str):
        if subset == "valid":
            for valid_sub_split in self.cfg.dataset.valid_subset.split(","):
                self.task.load_dataset(valid_sub_split, combine=False, epoch=1)
            return self.trainer.get_valid_iterator("valid").next_epoch_itr(shuffle=False, set_dataset_epoch=False)
        elif subset == "train":
            epoch_itr = self.trainer.get_train_iterator(
                epoch=1, load_dataset=True, disable_iterator_cache=self.task.has_sharded_data("train")
            )
            return epoch_itr.next_epoch_itr(fix_batches_to_gpus=self.cfg.distributed_training.fix_batches_to_gpus, shuffle=(
                epoch_itr.next_epoch_idx > self.cfg.dataset.curriculum))
        else:
            raise NotImplementedError(f"subset: {subset}")

    def extract_features(self):
        for dataset_path, checkpoint_path in tqdm(zip(self.datasets_paths, self.checkpoints_paths), total=len(self.datasets_paths)):
            self.build_stack(dataset_path, checkpoint_path)

            dataset_name = self._dataset_name(dataset_path)
            for subset in self.subsets:
                print(f"------------------------ {dataset_name}: {subset} ------------------------")
                subset_iterator = self.get_iterator(subset)

                features, labels = self._extract_features(subset_iterator, dataset_name, subset)
                # features = features.cpu()
                # labels = labels.cpu()

                self.dataset_instances[dataset_name][subset]["features"] = features
                self.dataset_instances[dataset_name][subset]["labels"] = labels

    @Cachable.cache()
    def _extract_features(self, iterator, dataset_name, subset) -> torch.TensorType:
        features = []
        labels = []
        c_n_batches = 0
        with torch.no_grad():
            for sample in iterator:
                if self.max_n_batches is not None and c_n_batches >= self.max_n_batches:
                    break
                net_input = move_to(sample['net_input'], device)
                target = move_to(sample['target'], device)

                output_tok, _ = self.model.extract_features(**net_input)
                output = self._pool(self.pooling_method, output_tok, input_tok=net_input)
                features.append(output)
                labels.append(target)
                c_n_batches += 1

        return torch.concat(features, dim=0), torch.concat(labels, dim=0)

    def fit_transform(self, **kwargs):
        return self.methods[self.method](**kwargs)

    def fit_transform_mds(self, n_components: int):
        return self.mds_transforms[self.cfg.model.fit_policy]

    def fit_transform_mds_seperate(self):
        pass

    def fit_transform_mds_grouped(self):
        pass

    def _fit_mds(self):
        pass

    def _transform_umap(self):
        pass

    def fit_transform_umap(self, n_neighbors: List[int], min_dists: List[float], max_repeat: int):
        return self.umap_transforms[self.cfg.model.fit_policy](n_neighbors, min_dists, max_repeat)

    def fit_transform_umap_separate(self, n_neighbors, min_dists, max_repeat):
        features = self._concat_dataset_instances(self.dataset_instances, "features").cpu()
        ground_features = self._concat_dataset_instances(
            {self.ground_dataset_name: self.dataset_instances[self.ground_dataset_name]}, "features").cpu()

        for n_neighbor in n_neighbors:
            for min_dist in min_dists:
                for i_repeat in range(max_repeat):
                    logger.info(f"umap[{i_repeat}]: n_neighbor:{n_neighbor}, min_dist:{min_dist}")
                    logger.info(f"fitting")
                    self._fit_umap(ground_features, n_neighbor=n_neighbor, min_dist=min_dist)
                    logger.info(f"transforming")
                    embeddings_full = self._transform_umap(features)
                    embeddings, ground_embeddings = self._separate_ground(embeddings_full)
                    logger.info(f"plotting")
                    self.plot(embeddings, ground_embeddings, n_neighbor=n_neighbor, min_dist=min_dist, i_repeat=i_repeat)

    def fit_transform_umap_grouped(self, n_neighbors, min_dists, max_repeat):
        features = self._concat_dataset_instances(self.dataset_instances, "features").cpu()

        for n_neighbor in n_neighbors:
            for min_dist in min_dists:
                for i_repeat in range(max_repeat):
                    logger.info(f"umap[{i_repeat}]: n_neighbor:{n_neighbor}, min_dist:{min_dist}")
                    logger.info(f"fitting")
                    self._fit_umap(features, n_neighbor=n_neighbor, min_dist=min_dist)
                    logger.info(f"transforming")
                    embeddings_full = self._transform_umap(features)
                    embeddings, ground_embeddings = self._separate_ground(embeddings_full)
                    logger.info(f"plotting")
                    torch.save(embeddings, self.log_dir.joinpath(f"[{i_repeat}]embeddings-{n_neighbor}-{min_dist}.pt"))
                    torch.save(ground_embeddings, self.log_dir.joinpath(f"[{i_repeat}]ground_embeddings-{n_neighbor}-{min_dist}.pt"))
                    self.plot(embeddings, ground_embeddings, n_neighbor=n_neighbor, min_dist=min_dist, i_repeat=i_repeat)

    def _fit_umap(self, features: torch.TensorType, n_neighbor: int, min_dist: float):
        self.reducer = umap.UMAP(n_neighbors=n_neighbor, min_dist=min_dist, low_memory=True)
        self.reducer.fit(features)

    def _transform_umap(self, features):
        return self.reducer.transform(features)

    def _separate_ground(self, obj):
        n_ground_dataset_items = 0
        for subset in self.subsets:
            n_ground_dataset_items += self.dataset_instances[self.ground_dataset_name][subset]["features"].shape[0]
        trimed = obj[:-n_ground_dataset_items, :]
        ground = obj[-n_ground_dataset_items:, :]

        return trimed, ground

    def _concat_dataset_instances(self, dataset_instances, tag):
        features_cumulative = []
        for dataset_instance in dataset_instances.values():
            for subset in self.subsets:
                features_cumulative.append(dataset_instance[subset][tag])

        return torch.concat(features_cumulative)

    def plot(self, embeddings, ground_embeddings, n_neighbor, min_dist, i_repeat):
        plt.gca().set_aspect('equal', 'datalim')
        plt.legend(fontsize=14, markerscale=2, facecolor='w')

        plt.title(
            f"UMAP projection of {len(self.dataset_instances.keys())} datasets, using '{self.ground_dataset_name}' as the ground dataset - number of neighbours:{n_neighbor}, min distance:{min_dist}", fontsize=15)

        self._plot(embeddings, ground_embeddings, self.dataset_instances, self.ground_dataset_name, n_neighbor, min_dist, i_repeat)

    def _plot(self, embeddings, ground_embeddings, dataset, ground_dataset, n_neighbor, min_dist, i_repeat):
        dataset_names = []
        subsets = []
        for dataset_name in dataset.keys():
            if dataset_name != ground_dataset:
                num_samples = 0
                for subset in self.subsets:
                    num_samples += dataset[dataset_name][subset]["features"].shape[0]
                    subsets += [subset] * dataset[dataset_name][subset]["features"].shape[0]
                dataset_names += [dataset_name] * num_samples
        data = {
            "x": embeddings[:, 0].tolist(),
            "y": embeddings[:, 1].tolist(),
            "subset": subsets,
            "dataset": dataset_names
        }
        df = pd.DataFrame(data=data)

        ground_dataset_names = [ground_dataset] * ground_embeddings.shape[0]
        ground_data = {
            "x": ground_embeddings[:, 0].tolist(),
            "y": ground_embeddings[:, 1].tolist(),
            "dataset": ground_dataset_names
        }
        ground_df = pd.DataFrame(data=ground_data)
        sns.kdeplot(x=ground_df.x, y=ground_df.y, cmap="light:b", shade=True, bw_adjust=.5)

        sns.scatterplot(data=df, x="x", y="y", hue="dataset", style="subset", alpha=0.85)
        plt.tight_layout()
        plt.savefig(self.log_dir.joinpath(f"[{i_repeat}]{n_neighbor}-{min_dist}.png"))
        plt.clf()

    def _pool(self, pooling: str, output_tok: torch.Tensor, **kwargs):
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

    def _dataset_name(self, dataset_path: str):
        return dataset_path.rstrip("/").split("/")[-2].lower()


def main(cfg: DictConfig) -> None:
    worker = Worker(cfg)

    worker.extract_features()
    n_neighbors = [int(item) for item in cfg.model.umap_n_neighbors.split(",")]
    min_dists = [float(item) for item in cfg.model.umap_min_dists.split(",")]
    worker.fit_transform(n_neighbors=n_neighbors, min_dists=min_dists, max_repeat=cfg.model.max_repeat)


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()

    parser.add_argument(
        "--method",
        type=str,
        default="",
        choices=["umap", "mds"],
        help="method(algorithm) of dimensionality reduction",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="dir path to store plots",
    )

    parser.add_argument(
        "--pooling_method",
        type=str,
        default="avg",
        help="Method of representing a sentence via tokens",
    )

    parser.add_argument(
        "--fit-policy",
        type=str,
        default="grouped",
        choices=["grouped", "separate"],
        help="Method of fitting the umap. grouped: fits all the embs concatenated. separate: fits each dataset emb separatly",
    )

    parser.add_argument(
        "--plot-policy",
        type=str,
        default="",
        help="plot on pre-trained model or on fine-tuned models",
    )

    parser.add_argument(
        "--max-repeat",
        type=int,
        default="",
        help="maximum num of times to plot with the same configs",
    )


    parser.add_argument(
        "--umap-n-neighbors",
        type=str,
        default="",
        help="number of neighbors for UMAP, seperated by comma",
    )

    parser.add_argument(
        "--umap-min-dists",
        type=str,
        default="",
        help="min distances for UMAP, seperated by comma",
    )

    parser.add_argument(
        "--umap-datasets",
        type=str,
        default="",
        help="list of datasets to plot on, comma separate",
    )

    parser.add_argument(
        "--umap-checkpoints",
        type=str,
        default="",
        help="list of checkpoints of each dataset, comma separate",
    )

    parser.add_argument(
        "--ground-dataset",
        type=str,
        default="",
        help="the name of the dataset for which to plot the contoures (pre-training dataset)",
    )

    parser.add_argument(
        "--n-batches",
        type=int,
        default=None,
        help="the number of consecutive batches to get from each dataset subset",
    )

    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
