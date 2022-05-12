# to download
# wget https://mystic.the-eye.eu/public/AI/pile/train/00.jsonl.zst
# to unzip -> jsonl
# unzstd 00.jsonl.zst


from pathlib import Path

from tqdm import tqdm
import argparse
import json

import random
import jsonlines
import numpy as np


def extract_json(in_path: str, out_path: str, num_shards: float, train_ratio: float):
    shards = [open(f"{out_path}.train.{shard}.tokens", "w") for shard in range(num_shards+1)]
    with open(in_path, "r") as in_file, open(f"{out_path}.val.tokens", "w") as out_val_file:
        for line in tqdm(in_file):
            line_text = json.loads(line)['text']
            prob_train = random.uniform(0, 1)
            if prob_train < train_ratio:
                prob_shard = np.random.randint(0, num_shards+1)
                shards[prob_shard].write(line_text)
            else:
                out_val_file.write(line_text)

    for shard_file in shards:
        shard_file.close()


parser = argparse.ArgumentParser(description='Extract a .zst file')
parser.add_argument('in_path', type=str,
                    help='input file (.zst) path')
parser.add_argument('--out_path', type=str,
                    help='output file path')
parser.add_argument('--num_shards', type=int, default=10,
                    help='number of shards')
parser.add_argument('--train_ratio', type=float, default=0.9,
                    help='portion of the set to keep')

args = parser.parse_args()

extract_json(args.in_path, args.out_path, args.num_shards, args.train_ratio)
print("done")
