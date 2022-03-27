# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from re import X
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from fairseq.fairseq.data.boosted_monolingual_dataset import BoostedMonolingualDataset
from fairseq.fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.fairseq.tasks import language_modeling
import torch
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    data_utils,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask
from omegaconf import II

logger = logging.getLogger(__name__)


@dataclass
class BoostedLanguageModelingConfig(LanguageModelingConfig):
    previous_lms: str = field(
        default="",
        metadata={
            "help": "comma-separated list of previous LM directories to boost on"},
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
        super().__init__(args,dictionary, output_dictionary, targets)
        model_paths = args.previous_lms.split(",")
        # TODO: check loading up
        self.weak_models = [TransformerLanguageModel.from_pretrained(x, "checkpoint_best.pt").models[0] for x in model_paths]


    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        super().load_dataset(split,epoch,combine)        

        self.datasets[split] = BoostedMonolingualDataset(self.datasets[split])