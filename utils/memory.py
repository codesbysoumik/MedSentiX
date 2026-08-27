"""Small, explicit helpers for releasing unreferenced PyTorch memory."""

from __future__ import annotations

import gc

import torch


def _accelerator_available() -> str | None:
    """Return 'cuda', 'mps', or None — whichever backend is actually active."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return None


def cleanup_memory() -> None:
    """Collect unreachable Python objects and release unused accelerator cache.

    Call this only after the caller has deleted references to completed models,
    training objects, loaders, and temporary tensors. It never frees live
    tensors or changes model state.
    """
    gc.collect()
    backend = _accelerator_available()
    if backend == "cuda":
        torch.cuda.empty_cache()
        # Releases CUDA IPC memory handles left behind by DataLoader worker
        # processes (num_workers > 0). Cheap no-op if nothing to release, but
        # matters most on setups running many sequential loaders — like the
        # baseline/MedSentiX training loop cycling through 7+ models.
        torch.cuda.ipc_collect()
    elif backend == "mps":
        torch.mps.empty_cache()


def report_memory(label: str = "") -> dict[str, float]:
    """Print and return current memory usage in GB, for spotting leaks between runs.

    Cheap enough to call between baselines/variants without materially
    affecting timing. Prints nothing and returns an empty dict if no
    accelerator is available (CPU-only runs still get gc.collect() from
    cleanup_memory(), just no memory figures here).
    """
    backend = _accelerator_available()
    if backend != "cuda":
        return {}
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}GPU mem — allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB | peak: {peak:.2f} GB")
    return {"allocated_gb": allocated, "reserved_gb": reserved, "peak_gb": peak}
