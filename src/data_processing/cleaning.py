"""Validation and numerical cleaning for raw NTN artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np


def clean_numeric_array(values: Any, clip_limit: float = 1e6) -> np.ndarray:
    """Return a finite float32 array, replacing invalid values deterministically."""
    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = np.asarray(array, dtype=np.float64)
    finite_values = array[np.isfinite(array)]
    replacement = float(np.median(finite_values)) if finite_values.size else 0.0
    array = np.nan_to_num(array, nan=replacement, posinf=clip_limit, neginf=-clip_limit)
    return np.clip(array, -clip_limit, clip_limit).astype(np.float32)


def z_score(values: Any, epsilon: float = 1e-6) -> np.ndarray:
    """Clean and standardize values, returning zeros for constant vectors."""
    array = clean_numeric_array(values)
    standard_deviation = float(array.std())
    if standard_deviation < epsilon:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - array.mean()) / standard_deviation).astype(np.float32)