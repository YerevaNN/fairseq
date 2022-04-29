# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from dataclasses import dataclass, field

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion, CrossEntropyCriterionConfig
from fairseq.data.boosted_monolingual_dataset import BoostedMonolingualDataset, WeakModels

import torch
from torch.nn.parameter import Parameter

from fairseq.models import register_model
from fairseq.models.transformer_lm import TransformerLanguageModel, TransformerLanguageModelConfig


@dataclass
class WithShrinkTransformerLanguageModelConfig(TransformerLanguageModelConfig):
    net_shrinkage: float = field(
        default=1.0, metadata={"help": "initial shrinkage coefficient of the current network"}
    )


@register_model("with_shrink_transformer_lm", dataclass=WithShrinkTransformerLanguageModelConfig)
class WithShrinkTransformerLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return out

    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        model.shrinkage = Parameter(torch.tensor(args.net_shrinkage, requires_grad=True, device="cuda"))

        return model


@register_criterion("boosted_cross_entropy", dataclass=CrossEntropyCriterionConfig)
class BoostedCrossEntropyCriterion(CrossEntropyCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task, sentence_avg)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, net_inner_states = model(**sample["net_input"])

        train_dataset: BoostedMonolingualDataset = self.task.datasets['train']
        if not hasattr(model, "shrinkage"):
            model.shrinkage = train_dataset.prev_lm_shrinkage

        if WeakModels.weak_models:
            sample['boosted_logits'] = train_dataset.get_batch_boosted_logits(
                sample['net_input']['src_tokens'])

            boosted_output = train_dataset.boost(sample['boosted_logits'].detach(), net_output,
                                                 shrinkage=model.shrinkage)

            output = (boosted_output, net_inner_states)
        else:
            output = (net_output, net_inner_states)

        loss, _ = self.compute_loss(model, output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "shrinkage_model": model.shrinkage
        }

        logging_shrinkage = train_dataset.get_weak_learners_shrinkages()
        logging_output.update(logging_shrinkage)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        metrics.log_scalar(
            "shrinkage_model", logging_outputs[-1]["shrinkage_model"].item()
        )

        for key in logging_outputs[-1]:
            if "shrinkage" in key:
                metrics.log_scalar(
                    key, logging_outputs[-1][key]
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
