# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Callable, Optional

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class SingleDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
    ) -> None:
        """
        A simplified dataloader that works with a single dataset.
        
        Args:
            dataset (Dataset): The dataset to load data from
            batch_size (int): How many samples per batch to load
            num_workers (int): How many subprocesses to use for data loading
            shuffle (bool): Whether to shuffle the data at every epoch
            pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory
            drop_last (bool): If True, drop the last incomplete batch
            collate_fn (callable): Merges a list of samples to form a mini-batch
            worker_init_fn (callable): If not None, this will be called on each worker subprocess
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self._iterator = None
        self._current_epoch = 0

    def __len__(self):
        if not self.drop_last:
            return math.ceil(len(self.dataset) / self.batch_size)
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        self._iterator = iter(self.get_loader(self._current_epoch))
        return self

    def __next__(self):
        if self._iterator is None:
            raise TypeError(f"{type(self).__name__} object is not an iterator")
        try:
            return next(self._iterator)
        except StopIteration:
            self._current_epoch += 1
            raise

    def get_loader(self, epoch: int) -> DataLoader:
        """
        Returns a DataLoader for the given epoch.
        
        Args:
            epoch (int): The current epoch number
        """
        # Set epoch for dataset if it supports it
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        elif hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch

        # Create sampler based on distributed training status
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            sampler = torch.utils.data.RandomSampler(self.dataset) if self.shuffle else torch.utils.data.SequentialSampler(self.dataset)
        else:
            sampler = DistributedSampler(self.dataset, shuffle=self.shuffle)
            sampler.set_epoch(epoch)

        # Create batch sampler
        batch_sampler = BatchSampler(sampler, self.batch_size, drop_last=self.drop_last)

        # Return dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_fn,
            worker_init_fn=self.worker_init_fn,
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Manually set the epoch number. This is useful when you want to
        start from a specific epoch.
        
        Args:
            epoch (int): The epoch number to set
        """
        self._current_epoch = epoch
