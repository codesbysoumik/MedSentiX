"""Device and reproducibility utilities for the MedSentiX project.

Every notebook and training module imports from this file so CUDA, Apple MPS,
and CPU execution are selected consistently without hardcoding a backend.
"""

from __future__ import annotations

import os
import platform
import random
from pathlib import Path

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


def get_amp_dtype(device: torch.device) -> torch.dtype:
    """Pick the mixed-precision dtype the current GPU can actually accelerate.

    bf16 tensor cores require Ampere (SM80) or newer — e.g. your local RTX
    5070 Ti (Blackwell). Kaggle's T4 (Turing, SM75) has no bf16 tensor core
    support. Newer PyTorch's is_bf16_supported() defaults to counting
    *software-emulated* bf16 as "supported" (via its including_emulation
    flag), which returns True even on a T4 — so calling it without that
    flag silently picks the slow, unaccelerated path instead of erroring.
    We explicitly ask for hardware-accelerated support only, falling back
    to the plain call on older PyTorch versions that don't have the
    parameter at all.

    T4 does have fast fp16 tensor cores, so we fall back to fp16 there.
    fp16 needs loss scaling (see torch.cuda.amp.GradScaler) since its
    exponent range is much narrower than bf16's — bf16 doesn't need a
    scaler at all.
    """
    if device.type != "cuda":
        return torch.float32
    try:
        bf16_ok = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        # Older PyTorch — no including_emulation parameter, plain call only
        # checks hardware capability already.
        bf16_ok = torch.cuda.is_bf16_supported()
    return torch.bfloat16 if bf16_ok else torch.float16


def running_on_kaggle() -> bool:
    """True when executing inside a Kaggle notebook session.

    Used to gate the extra plain-text progress prints in the training
    loops: Kaggle's background/commit log export doesn't reliably capture
    tqdm's carriage-return progress updates, so those loops print explicit
    lines there. Locally (VS Code, JupyterLab, etc.) tqdm already renders
    live, so the extra prints are just redundant clutter — skip them there.
    """
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle").exists()


__all__ = ["RANDOM_SEED", "get_device", "set_seed", "default_num_workers", "get_amp_dtype", "running_on_kaggle"]