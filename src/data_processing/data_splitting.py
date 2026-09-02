"""Temporal train, validation, and test partitioning."""

from __future__ import annotations

from typing import Dict

import numpy as np


def chronological_split(scenario_ids: np.ndarray, train_fraction: float = 0.70,
                        validation_fraction: float = 0.15) -> Dict[str, np.ndarray]:
    """Split sorted scenario IDs without temporal leakage."""
    ordered_indices = np.argsort(scenario_ids)
    total = len(ordered_indices)
    if total < 3:
        raise ValueError("At least three scenarios are required for train/validation/test splitting")
    train_end = max(1, int(total * train_fraction))
    validation_end = max(train_end + 1, int(total * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, total - 1)
    return {
        "train": ordered_indices[:train_end],
        "validation": ordered_indices[train_end:validation_end],
        "test": ordered_indices[validation_end:],
    }