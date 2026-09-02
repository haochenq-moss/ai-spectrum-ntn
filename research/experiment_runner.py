"""Offline, bounded AutoResearch loop for spectrum allocation experiments."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from src.models.supervised_learning import evaluate_supervised, train_supervised


HYPOTHESES = [
    (1e-4, "Lower learning rate may improve held-out imitation accuracy."),
    (3e-4, "Moderate learning rate may improve held-out imitation accuracy."),
    (1e-3, "Baseline learning rate for supervised allocation."),
    (3e-3, "Higher learning rate may converge faster within the trial budget."),
]


def _best_score(results_file: Path) -> tuple[float, float]:
    if not results_file.exists():
        return float("inf"), float("-inf")
    with results_file.open(newline="") as source:
        return min(
            ((float(row["test_mse"]), -float(row["test_spectral_efficiency"]))
             for row in csv.DictReader(source, delimiter="\t") if row["status"] == "keep"),
            default=(float("inf"), float("-inf")),
        )


def _append_result(results_file: Path, row: dict[str, Any]) -> None:
    write_header = not results_file.exists()
    with results_file.open("a", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=row.keys(), delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def launch_autoresearch_loop(duration_seconds: float = 28800, gpu_id: int = 0,
                              program_file: str = "research/program.md",
                              results_file: str = "research/model_results_v2.tsv",
                              learnings_file: str = "research/model_learnings_v2.md",
                              num_scenarios: int = 3, max_trials: int = 4,
                              epochs: int = 100) -> dict[str, Any]:
    """Run bounded supervised-model trials and log held-out evaluation metrics."""
    if not Path(program_file).is_file():
        raise FileNotFoundError(f"Research program does not exist: {program_file}")
    results_path, learnings_path = Path(results_file), Path(learnings_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    best_key = _best_score(results_path)
    started, trial_index, completed = time.monotonic(), 0, []
    while time.monotonic() - started < duration_seconds and trial_index < max_trials:
        learning_rate, hypothesis = HYPOTHESES[trial_index % len(HYPOTHESES)]
        model = train_supervised(
            epochs=epochs,
            device=f"cuda:{gpu_id}",
            training_config={"learning_rate": learning_rate, "batch_size": 32},
            model_path=f"research/experiment_logs/trial_{trial_index:03d}/supervised_policy.pt",
            max_train_samples=num_scenarios,
            random_seed=42,
        )
        metrics = evaluate_supervised(model)
        efficiency, latency = metrics["test_spectral_efficiency"], metrics["inference_latency_ms"]
        candidate_key = (metrics["test_mse"], -efficiency)
        status = "keep" if candidate_key < best_key and latency <= 10.0 else "discard"
        if status == "keep":
            best_key = candidate_key
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trial": trial_index,
            "learning_rate": learning_rate,
            "test_mse": f"{metrics['test_mse']:.8f}",
            "test_spectral_efficiency": f"{efficiency:.6f}",
            "inference_latency_ms": f"{latency:.6f}",
            "status": status,
            "selection_rule": "lowest_test_mse_then_highest_spectral_efficiency",
            "description": hypothesis,
        }
        _append_result(results_path, row)
        with learnings_path.open("a") as learnings:
            learnings.write(f"- Trial {trial_index}: lr={learning_rate:g} {status}; test MSE={metrics['test_mse']:.8f}, test efficiency={efficiency:.3f}, inference={latency:.3f} ms. Selected by lowest test MSE, then efficiency. {hypothesis}\n")
        completed.append(row)
        trial_index += 1
        if duration_seconds <= 1:
            break
    return {
        "trials": len(completed),
        "num_scenarios": num_scenarios,
        "epochs_per_trial": epochs,
        "results_file": str(results_path),
        "learnings_file": str(learnings_path),
        "best_test_mse": best_key[0],
        "best_test_spectral_efficiency": -best_key[1],
    }