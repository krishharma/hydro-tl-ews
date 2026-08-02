"""Shared DataLoader construction with device-aware defaults."""
from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from .device import training_device


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader with ``pin_memory`` enabled on CUDA."""
    kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
    }
    if training_device() == "cuda":
        kwargs["pin_memory"] = True
    return DataLoader(dataset, **kwargs)
