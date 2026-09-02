"""Formal regression tests for held-out model trials and Slurm configuration."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from research import experiment_runner
from src.models.supervised_learning import evaluate_supervised, train_supervised


def _write_teacher_dataset(path: Path) -> None:
    states = np.linspace(0, 1, 4 * 256, dtype=np.float32).reshape(4, 256)
    actions = np.full((4, 16, 32), 1 / 512, dtype=np.float32)
    with h5py.File(path, "w") as output:
        output.create_dataset("states", data=states)
        output.create_dataset("actions", data=actions)


def test_supervised_training_uses_train_split_and_evaluates_held_out_data(tmp_path: Path) -> None:
    teacher_path = tmp_path / "teacher.h5"
    split_path = tmp_path / "splits.json"
    model_path = tmp_path / "policy.pt"
    _write_teacher_dataset(teacher_path)
    split_path.write_text(json.dumps({"train": [0, 1], "validation": [2], "test": [3]}))

    model = train_supervised(
        epochs=1,
        device="cpu",
        teacher_path=str(teacher_path),
        split_path=str(split_path),
        model_path=str(model_path),
        max_train_samples=1,
        random_seed=42,
    )
    metrics = evaluate_supervised(model, str(teacher_path), str(split_path))

    assert model_path.is_file()
    assert set(metrics) == {"test_mse", "test_spectral_efficiency", "inference_latency_ms"}
    assert metrics["test_mse"] >= 0
    assert metrics["test_spectral_efficiency"] > 0


def test_autoresearch_promotes_lowest_test_mse(monkeypatch, tmp_path: Path) -> None:
    program_path = tmp_path / "program.md"
    results_path = tmp_path / "results.tsv"
    learnings_path = tmp_path / "learnings.md"
    program_path.write_text("# test program\n")
    metrics = iter([
        {"test_mse": 0.4, "test_spectral_efficiency": 100.0, "inference_latency_ms": 1.0},
        {"test_mse": 0.2, "test_spectral_efficiency": 90.0, "inference_latency_ms": 1.0},
        {"test_mse": 0.3, "test_spectral_efficiency": 200.0, "inference_latency_ms": 1.0},
    ])
    monkeypatch.setattr(experiment_runner, "train_supervised", lambda **_: object())
    monkeypatch.setattr(experiment_runner, "evaluate_supervised", lambda _: next(metrics))

    result = experiment_runner.launch_autoresearch_loop(
        duration_seconds=600,
        program_file=str(program_path),
        results_file=str(results_path),
        learnings_file=str(learnings_path),
        max_trials=3,
        epochs=1,
    )
    rows = list(csv.DictReader(results_path.open(), delimiter="\t"))

    assert result["best_test_mse"] == 0.2
    assert [row["status"] for row in rows] == ["keep", "keep", "discard"]
    assert "Selected by lowest test MSE" in learnings_path.read_text()


def test_autoresearch_slurm_profile_is_gpu_bounded() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "autoresearch.sbatch").read_text()
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --time=00:10:00" in script
    assert "--num-scenarios 3 --max-trials 4 --epochs 100" in script