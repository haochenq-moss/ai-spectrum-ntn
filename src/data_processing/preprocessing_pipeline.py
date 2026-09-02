"""CPU-only orchestration from raw NTN artifacts to learning states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .cleaning import clean_numeric_array
from .data_splitting import chronological_split
from .feature_engineering import (
    encode_channel, encode_interference, encode_jamming, encode_topology, encode_traffic,
)
from .state_construction import NetworkState, build_context_features


def _load_csi(path: Path) -> dict[str, dict[str, Any]]:
    with h5py.File(path, "r") as source:
        return {
            link_name: {
                "channel_matrix": source[link_name]["channel_matrix"][()],
                **dict(source[link_name].attrs),
            }
            for link_name in source.keys()
        }


def _normalize_from_training(states: np.ndarray, train_indices: np.ndarray) -> np.ndarray:
    train_states = states[train_indices]
    mean = train_states.mean(axis=0)
    standard_deviation = train_states.std(axis=0)
    return ((states - mean) / np.where(standard_deviation < 1e-6, 1.0, standard_deviation)).astype(np.float32)


def process_all(input_dir: str = "data/raw/", output_dir: str = "data/processed/",
                feature_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create 256-dimensional normalized states and temporal split metadata."""
    del feature_config
    raw_dir, processed_dir = Path(input_dir), Path(output_dir)
    manifest = json.loads((raw_dir / "dataset_manifest.json").read_text())
    states, scenario_ids = [], []

    for entry in manifest["manifests"]:
        if "error" in entry:
            continue
        scenario_id = int(entry["scenario_id"])
        files = entry["files"]
        scenario = json.loads(Path(files["scenario_json"]).read_text())
        csi_links = _load_csi(Path(files["csi_h5"]))
        traffic = pd.read_csv(files["traffic_csv"])
        interference = clean_numeric_array(np.load(files["interference_npy"]))
        jamming = [np.load(path) for path in files.get("jamming_npy", [])]
        state = NetworkState(
            topology_embedding=encode_topology(scenario),
            channel_state=encode_channel(csi_links),
            traffic_features=encode_traffic(traffic),
            interference_features=encode_interference(interference),
            jammer_profile=encode_jamming(jamming),
            context_features=build_context_features(scenario, scenario_id, len(csi_links)),
        )
        states.append(state.to_vector())
        scenario_ids.append(scenario_id)

    if not states:
        raise ValueError(f"No successful scenario entries found in {raw_dir / 'dataset_manifest.json'}")
    raw_states = np.stack(states).astype(np.float32)
    scenario_ids_array = np.asarray(scenario_ids, dtype=np.int32)
    splits = chronological_split(scenario_ids_array)
    normalized_states = _normalize_from_training(raw_states, splits["train"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_dir / "learning_states.npz",
        states=normalized_states,
        raw_states=raw_states,
        scenario_ids=scenario_ids_array,
    )
    split_payload = {name: indices.tolist() for name, indices in splits.items()}
    (processed_dir / "splits.json").write_text(json.dumps(split_payload, indent=2))
    summary = {"num_states": len(states), "state_dimension": 256, "output_dir": str(processed_dir), "splits": split_payload}
    (processed_dir / "processing_summary.json").write_text(json.dumps(summary, indent=2))
    return summary