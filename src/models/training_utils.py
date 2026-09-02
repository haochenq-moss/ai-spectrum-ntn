"""Shared CPU/GPU-safe helpers for Step 4 model training."""

from __future__ import annotations

from pathlib import Path

import torch


def resolve_device(requested_device: str) -> torch.device:
    """Resolve a requested device, falling back to CPU when CUDA is unavailable."""
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested_device)
    return torch.device("cpu")


def default_model_path(name: str) -> Path:
    """Return and create the standard model artifact location."""
    destination = Path("data/models")
    destination.mkdir(parents=True, exist_ok=True)
    return destination / f"{name}.pt"