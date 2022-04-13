# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field

from fairseq.data.boosted_monolingual_dataset import (BoostedMonolingualDataset, WeakModels)
from fairseq.models.transformer_lm import TransformerLanguageModel
import torch
from fairseq.data import MonolingualDataset
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask

logger = logging.getLogger(__name__)


@dataclass
class BoostedLanguageModelingConfig(LanguageModelingConfig):
    previous_lms: str = field(
        default="",
        metadata={
            "help": "comma-separated list of previous LM directories to boost on"},
    )
    alpha: float = field(
        default=1.0, metadata={"help": "shrinkage rate of previous_lms: -1 parameterize, -2 will linearly decrease"}
    )
    beta: float = field(
        default=1.0, metadata={"help": "shrinkage rate of current model: -1 parameterize, -2 will linearly increase"}
    )
    model_better_init: bool = field(
        default=False, metadata={"help": "initialize the model with the previous lms params"}
    )
    logits_cache_dir: str = field(
        default="",
        metadata={
            "help": "dir to cache the logits, if not specified will not be cached"},
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
        # self.alpha = torch.tensor([args.alpha], requires_grad=True)
        # self.beta = torch.tensor([args.beta], requires_grad=True)
        self.alpha = args.alpha
        self.beta = args.beta
        self.model_better_init = args.model_better_init
        self.logits_cache_dir = args.logits_cache_dir

        model_paths = args.previous_lms.split(",")

        if WeakModels.weak_models is None:
            WeakModels.weak_models = torch.nn.ModuleList().extend(
                TransformerLanguageModel.from_pretrained(
                    x, checkpoint_file='checkpoint_best.pt', data_name_or_path=args.data).models[0]
                for x in model_paths)
            WeakModels.weak_models = WeakModels.weak_models.to(self.device)

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        super().load_dataset(split, epoch, combine)

        self.datasets[split] = BoostedMonolingualDataset(
            self.datasets[split], self.device, alpha=self.alpha, beta=self.beta, model_better_init=self.model_better_init, logits_cache_dir=self.logits_cache_dir)

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported language modeling target: {}".format(target)
                )

        if self.model_better_init:
            for model_param, weak_model_param in zip(model.parameters(), WeakModels.weak_models[-1].parameters()):
                model_param.data = weak_model_param.data

        return model
