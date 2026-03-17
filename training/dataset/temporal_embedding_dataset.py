# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset that loads saved temporal-matching embedding pairs for head-only training."""

import glob
import os
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class TemporalEmbeddingDataset(Dataset):
    """
    Loads temporal matching pairs from a directory of .pt files.
    Each .pt file contains a list of pair dicts (key_tokens, key_centroids, key_ids,
    query_tokens, query_centroids, query_ids, child_to_parent, match_targets).
    """

    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.pairs: List[Dict[str, Any]] = []
        self._load_pairs()

    def _load_pairs(self) -> None:
        pattern = os.path.join(self.dir_path, "*.pt")
        files = sorted(glob.glob(pattern))
        for path in files:
            data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(data, list):
                self.pairs.extend(data)
            else:
                self.pairs.append(data)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.pairs[idx]
