# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion, CrossEntropyCriterionConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks.diffusion_language_modeling import DiffusionLanguageModelingTask
from omegaconf import II


@register_criterion("diffusion_cross_entropy", dataclass=CrossEntropyCriterionConfig)
class DiffusionCrossEntropyCriterion(CrossEntropyCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if "net_output" not in sample:
            net_output = model(**sample["net_input"])
        else:
             net_output = sample["net_output"]

        if self.task.n_diffusions > 0:
            # diffused_emb = self.project(model, net_output)
            with torch.no_grad():
                lprobs = model.get_normalized_probs(net_output, log_probs=False).to(torch.float16)
                diffused_emb = lprobs @ model.decoder.embed_tokens.weight
                sample["net_input"]["diffused_emb"] = diffused_emb

            net_output = model(**sample["net_input"])
            sample["net_output"] = net_output
        sample["diffusion_state"] += 1

        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output


    # def project(self, model, net_output):
        # lprobs = model.get_normalized_probs(net_output, log_probs=False).to(torch.float16)
        # return lprobs @ model.decoder.embed_tokens.weight

    # TODO - later implementation (optimized)
    # def partial_projection(probs, embedding_weight, n: int):
    #     max_elem, max_elem_ind = torch.topk(probs, n)
    #     partial_embed = embedding_weight[max_elem_ind]
    #     return max_elem @ partial_embed

    # print(partial_projection(probs,embedding.weight, 4))
