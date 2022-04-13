import logging
from asyncio.log import logger
import torch
import os
from pathlib import Path
import hashlib
from dataclasses import dataclass

from . import BaseWrapperDataset, MonolingualDataset, data_utils

logger = logging.getLogger(__name__)


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
        # torch.nn.utils.rnn.pad_sequence([s["boosted_logits"][0] for s in samples], batch_first=True)
        "boosted_logits": torch.zeros((1))
    }


# Dataclass: the weak_models will be set form BoostedLanguageModelingTask
# we are not passing the weak_models as an argument to __init__
# because in multiGPU multiprocessing case it tries to pickle
# BoostedMonolingualDataset s properties, and it failes on pickling a fairseqModel
@dataclass
class WeakModels:
    weak_models = None


class BoostedMonolingualDataset(BaseWrapperDataset):
    def __init__(self, dataset: MonolingualDataset, device=None, **kwargs):
        super().__init__(dataset)
        self.__dict__.update(kwargs)

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not device else device
        self.device = device
        # /hdd/.cache/boosted_logits
        if self.logits_cache_dir:
            os.makedirs(self.logits_cache_dir, exist_ok=True)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def _hash(self, name):
        return hashlib.md5(bytes(name, encoding='utf-8')).hexdigest()

    def get_batch_boosted_logits(self, item_sources):
        if self.logits_cache_dir:
            item_sources_hash = self._hash(str(item_sources.cpu().detach().numpy())) + ".pt"
            batch_cache_path = self.logits_cache_dir.joinpath(item_sources_hash)

            if not batch_cache_path.is_file():
                boosted_logits, merged_inner_states = self.batch_boosted_logits(item_sources)
                logger.info(f"creating a cache at: {batch_cache_path}")
                # start = time.time()
                torch.save((boosted_logits.detach().cpu(), merged_inner_states), batch_cache_path)
                # end = time.time()
                # logger.info(f"created a cache, took: {end-start}s")
            else:
                logger.info(f"reading from cache: {batch_cache_path}")
                # start = time.time()
                boosted_logits, merged_inner_states = torch.load(batch_cache_path)
                # end = time.time()
                # logger.info(f"read a cache, took: {end-start}s")
        else:
            # logger.info(f"evaluating")
            # start = time.time()
            # boosted_logits, merged_inner_states = self.batch_boosted_logits(item_sources)
            # end = time.time()
            # logger.info(f"evaluated, took: {end-start}s")
            # return boosted_logits, merged_inner_states
            return self.batch_boosted_logits(item_sources)

    def batch_boosted_logits(self, item_sources):
        logits = None
        inner_states = None

        item_sources = item_sources.to(self.device)
        with torch.no_grad():
            for model in WeakModels.weak_models:
                model.eval()
                # model_logits, _ = model.extract_features(item_sources, encoder_out=None)
                model_logits, model_inner_states = model(item_sources, encoder_out=None)
                model_logits = model_logits.detach().cpu()
                if logits is not None:
                    # shrinkage = alpha
                    logits = self.boost(logits, model_logits, self.alpha)
                    inner_states = self.merge_inner_state(inner_states, model_inner_states)
                else:
                    logits = model_logits
                    inner_states = model_inner_states

        return logits, inner_states

    def merge_inner_state(self, inner_states, model_inner_states):
        for inner_state, model_inner_state in zip(inner_states['inner_states'], model_inner_states['inner_states']):
            inner_state = inner_state.to(model_inner_state.device)
            model_inner_state.data += inner_state.data

            del inner_state
        return model_inner_states

    def boost(self, logits, model_logits, shrinkage: float = 1.0):
        if logits.device != model_logits.device:
            if logits.device.type == "cuda":
                model_logits = model_logits.to(logits.device)
            elif model_logits.device.type == "cuda":
                logits = logits.to(model_logits.device)
 
        return logits + (shrinkage * model_logits)

    def collater(self, samples):
        return collate(
            samples,
            self.dataset.vocab.pad(),
            self.dataset.vocab.eos(),
            self.dataset.fixed_pad_length,
            self.dataset.pad_to_bsz,
        )
