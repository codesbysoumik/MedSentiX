"""Device and reproducibility utilities for the MedSentiX project.

Every notebook and training module imports from this file so CUDA, Apple MPS,
and CPU execution are selected consistently without hardcoding a backend.
"""

from __future__ import annotations

import os
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


__all__ = ["RANDOM_SEED", "get_device", "set_seed"]