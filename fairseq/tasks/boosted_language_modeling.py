# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field

from fairseq.data.boosted_monolingual_dataset import (BoostedMonolingualDataset, WeakModels)
from fairseq.models.transformer_lm import TransformerLanguageModel
import torch
from torch.nn.parameter import Parameter
from fairseq.data import MonolingualDataset
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask

logger = logging.getLogger(__name__)


@dataclass
class BoostedLanguageModelingConfig(LanguageModelingConfig):
    previous_lms: str = field(
        default="",
        metadata={"help": "comma-separated list of previous LM directories to boost on"},
    )
    logits_cache_dir: str = field(
        default="",
        metadata={"help": "dir to cache the logits, if not specified will not be cached"},
    )
    logits_reduction: str = field(
        default="",
        metadata={"help": "type of reduction of logits"}
    )
    prev_lm_shrinkage: float = field(
        default=1.0,
        metadata={"help": "default value for model shrinkage: tensor(1)"}
    )


@register_task("boosted_language_modeling", dataclass=BoostedLanguageModelingConfig)
class BoostedLanguageModelingTask(LanguageModelingTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary, targets)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logits_cache_dir = args.logits_cache_dir
        self.logits_reduction = args.logits_reduction
        self.prev_lm_shrinkage = torch.tensor(args.prev_lm_shrinkage, requires_grad=False)

        if args.previous_lms and WeakModels.weak_models is None:
            model_paths = args.previous_lms.split(",")
            WeakModels.weak_models = torch.nn.ModuleDict()
            for i, model_path in enumerate(model_paths):
                model = TransformerLanguageModel.from_pretrained(
                    model_path, checkpoint_file='checkpoint_best.pt', data_name_or_path=args.data).models[0]
                if not hasattr(model, "shrinkage"):
                    model.shrinkage = self.prev_lm_shrinkage

                model = model.to(self.device)
                model.shrinkage = model.shrinkage.to(self.device)
                WeakModels.weak_models.add_module(str(i), model)

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        super().load_dataset(split, epoch, combine)

        self.datasets[split] = BoostedMonolingualDataset(self.datasets[split], self.device,
                                                         logits_cache_dir=self.logits_cache_dir,
                                                         logits_reduction=self.logits_reduction,
                                                         prev_lm_shrinkage=self.prev_lm_shrinkage)

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported language modeling target: {}".format(target)
                )

        return model
