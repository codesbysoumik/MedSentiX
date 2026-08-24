"""Small, explicit helpers for releasing unreferenced PyTorch memory."""

from __future__ import annotations

import gc

import torch


def cleanup_memory() -> None:
    """Collect unreachable Python objects and release unused CUDA cache blocks.

    Call this only after the caller has deleted references to completed models,
    training objects, loaders, and temporary tensors. It never frees live
    tensors or changes model state.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

