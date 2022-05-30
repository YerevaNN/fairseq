# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field

from fairseq.data.diffusion_monolingual_dataset import DiffusionMonolingualDataset
from fairseq.models.transformer_lm import TransformerLanguageModel
import torch
from torch.nn.parameter import Parameter
from fairseq.data import MonolingualDataset
from fairseq.tasks import register_task
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask

logger = logging.getLogger(__name__)


@dataclass
class DiffusionLanguageModelingConfig(LanguageModelingConfig):
    n_diffusions: int = field(
        default=0,
        metadata={"help": "state of the diffusion process"},
    )


@register_task("diffusion_language_modeling", dataclass=DiffusionLanguageModelingConfig)
class DiffusionLanguageModelingTask(LanguageModelingTask):
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
        self.n_diffusions = args.n_diffusions
        # self.diffusion_state = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        super().load_dataset(split, epoch, combine)

        self.datasets[split] = DiffusionMonolingualDataset(self.datasets[split], self.device)

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported language modeling target: {}".format(target)
                )

        return model

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        # for i in range(self.n_diffusions+1):
            # if i != self.n_diffusions and self.n_diffusions !=0:
            #     retain_graph = True
            # else:
            #     retain_graph = False

        model.train()
        model.set_num_updates(update_num)
        while sample["diffusion_state"] < self.n_diffusions:
            # optimizer.zero_grad()
            with torch.autograd.profiler.record_function("forward"):
                with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                    loss, sample_size, logging_output = criterion(model, sample)
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)
        return loss, sample_size, logging_output
