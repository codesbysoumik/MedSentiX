"""Device and reproducibility utilities for the MedSentiX project.

Every notebook and training module imports from this file so CUDA, Apple MPS,
and CPU execution are selected consistently without hardcoding a backend.
"""

from __future__ import annotations

import os
import platform
import random

import numpy as np
import torch


RANDOM_SEED = 42


def get_device() -> torch.device:
    """Return the best available torch device and print the selected backend."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Selected device: CUDA ({torch.cuda.get_device_name(0)})")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Selected device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("Selected device: CPU")
    return device


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def default_num_workers(cap: int = 4) -> int:
    """Pick a DataLoader worker count that's safe for the current platform.

    Windows (and macOS, as of Python 3.8+) default to the 'spawn'
    multiprocessing start method: each worker gets a fresh interpreter and
    the dataset object is pickled and shipped across a pipe to it — no
    memory sharing. Linux defaults to 'fork', which shares the parent's
    memory via copy-on-write, so extra workers are nearly free by
    comparison. On a RAM-constrained laptop, several 'spawn' workers per
    loader (and several loaders alive across sequential baselines) can add
    up fast, so we cap harder there.

    `cap` is the ceiling for platforms where workers are cheap (Linux,
    Kaggle). Windows gets min(2, cpu_count) regardless of `cap`, since even
    2 spawn workers can meaningfully add up across the train/val/test
    loaders this project builds per baseline.
    """
    cpu_count = os.cpu_count() or 1
    if platform.system() == "Windows":
        return max(0, min(2, cpu_count))
    return max(0, min(cap, cpu_count - 1))


__all__ = ["RANDOM_SEED", "get_device", "set_seed", "default_num_workers"]