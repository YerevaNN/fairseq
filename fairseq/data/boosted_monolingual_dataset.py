import numpy as np
import torch

from typing import List

from . import BaseWrapperDataset, MonolingualDataset, data_utils


def collate(samples, pad_idx, eos_idx, fixed_pad_length=None, pad_to_bsz=None):
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(
                    data_utils.collate_tokens(
                        [s[key][i] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                        pad_to_length=fixed_pad_length,
                        pad_to_bsz=pad_to_bsz,
                    )
                )
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
                pad_to_length=fixed_pad_length,
                pad_to_bsz=pad_to_bsz,
            )

    src_tokens = merge("source")
    if samples[0]["target"] is not None:
        is_target_list = isinstance(samples[0]["target"], list)
        target = merge("target", is_target_list)
    else:
        target = src_tokens

    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(len(s["source"]) for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
        },
        "target": target,
        "boosted_logits": torch.stack([x['boosted_logits'] for x in samples], dim=0)
    }


class BoostedMonolingualDataset(BaseWrapperDataset):
    def __init__(self, dataset: MonolingualDataset, models=torch.nn.ModuleList):
        super().__init__(dataset)
        self.models = models
        # Implementation 2
        # self.cached_logits = []
        # for i in range(len(self)):
        #     item = super().__getitem__(index)
        #     item_source = item["source"]
        #     self.cached_logits.append(self.get_boosted_logits(item_source))

        # Implementation 2++
        # batch items, collate them, do batch processing

    # Implementation 1. Dynamically process every sample as it's called
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item_source = item["source"]
        item['boosted_logits'] = self.get_boosted_logits(item_source)
        return item

    # Implementation 2
    # def __getitem__(self, index):
    #     return self.cached_logits[index]

    def get_boosted_logits(self, item_source):
        logits = None
        for model in self.models:
            model_logits = model.extract_features(item_source.unsqueeze(0), None)
            if logits is not None:
                logits += model_logits
            else:
                logits = model_logits

    def collater(self, samples):
        return collate(
            samples,
            self.vocab.pad(),
            self.vocab.eos(),
            self.fixed_pad_length,
            self.pad_to_bsz,
        )
