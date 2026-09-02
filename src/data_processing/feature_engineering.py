"""Deterministic fixed-width feature encoders for NTN scenario artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cleaning import clean_numeric_array


def _resize(values: Any, size: int) -> np.ndarray:
    values = clean_numeric_array(values).ravel()
    if values.size == 0:
        return np.zeros(size, dtype=np.float32)
    if values.size == size:
        return values
    return np.interp(
        np.linspace(0, values.size - 1, size),
        np.arange(values.size),
        values,
    ).astype(np.float32)


def encode_topology(scenario: dict[str, Any]) -> np.ndarray:
    """Encode topology and atmospheric conditions into 64 features."""
    satellites = scenario.get("satellites", [])
    ground_stations = scenario.get("ground_stations", [])
    user_equipment = scenario.get("user_equipment", [])
    cells = scenario.get("terrestrial_cells", [])
    atmosphere = scenario.get("atmospheric_conditions", {})
    counts = [len(satellites), len(ground_stations), len(user_equipment), len(cells)]
    conditions = [
        scenario.get("coverage_radius_km", 0.0),
        atmosphere.get("rain_rate_mm_h", 0.0),
        atmosphere.get("temperature_c", 0.0),
        atmosphere.get("humidity_percent", 0.0),
    ]
    positions = []
    for node in satellites + ground_stations + user_equipment + cells:
        latitude = float(node.get("latitude", 0.0))
        longitude = float(node.get("longitude", 0.0))
        positions.extend([np.sin(np.deg2rad(latitude)), np.cos(np.deg2rad(latitude))])
        positions.extend([np.sin(np.deg2rad(longitude)), np.cos(np.deg2rad(longitude))])
    return _resize(counts + conditions + positions, 64)


def encode_channel(csi_links: dict[str, dict[str, Any]]) -> np.ndarray:
    """Compress CSI magnitudes and link metadata into 128 features."""
    receiver_profiles, transmitter_profiles, metadata = [], [], []
    for link in csi_links.values():
        magnitude = clean_numeric_array(link["channel_matrix"])
        receiver_profiles.append(magnitude.mean(axis=(1, 2)))
        transmitter_profiles.append(magnitude.mean(axis=(0, 2)))
        metadata.extend(
            [
                link.get("path_loss_db", 0.0),
                link.get("doppler_hz", 0.0),
                link.get("atmospheric_attenuation_db", 0.0),
                link.get("distance_m", 0.0),
            ]
        )
    if not receiver_profiles:
        return np.zeros(128, dtype=np.float32)
    profiles = np.concatenate(
        [np.mean(receiver_profiles, axis=0), np.mean(transmitter_profiles, axis=0)]
    )
    return _resize(np.concatenate([profiles, clean_numeric_array(metadata)]), 128)


def encode_traffic(traffic: pd.DataFrame) -> np.ndarray:
    """Summarize load, QoS, and class mix into 32 features."""
    numeric_columns = [
        "downlink_kbps", "uplink_kbps", "total_kbps",
        "qos_latency_target_ms", "qos_max_loss_percent",
    ]
    summary = []
    for column in numeric_columns:
        values = clean_numeric_array(traffic.get(column, []))
        summary.extend([values.mean() if values.size else 0.0, values.std() if values.size else 0.0])
    traffic_mix = traffic.get("traffic_type", pd.Series(dtype=str)).value_counts(normalize=True)
    summary.extend(float(traffic_mix.get(name, 0.0)) for name in ["real_time", "video_streaming", "bursty", "background"])
    summary.extend([float(len(traffic)), float(traffic.get("ue_id", pd.Series()).nunique())])
    return _resize(summary, 32)


def encode_interference(interference: Any) -> np.ndarray:
    """Summarize interference power and temporal variability into 8 features."""
    values = clean_numeric_array(interference)
    return np.asarray(
        [values.mean(), values.std(), values.min(), values.max(), *np.percentile(values, [10, 50, 90]), values.shape[0]],
        dtype=np.float32,
    )


def encode_jamming(jamming_signals: list[Any]) -> np.ndarray:
    """Summarize aggregate jamming energy and frequency occupancy into 16 features."""
    if not jamming_signals:
        return np.zeros(16, dtype=np.float32)
    power = np.sum([clean_numeric_array(signal) ** 2 for signal in jamming_signals], axis=0)
    active_fraction = float(np.mean(power > 1e-12))
    features = [power.mean(), power.std(), power.max(), active_fraction, len(jamming_signals)]
    features.extend(np.percentile(power, [5, 25, 50, 75, 95]))
    features.extend(power.mean(axis=1))
    return _resize(features, 16)