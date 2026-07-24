"""Utilities shared by training scripts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ml.preprocessing import SEED, set_seed

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute the project-standard scalar classification metrics."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(precision), 6),
        "macro_recall": round(float(recall), 6),
        "macro_f1": round(float(f1), 6),
    }


def write_run_metrics(
    *,
    task: str,
    model: str,
    dataset: str,
    split: str,
    metrics: dict[str, float],
    artifact_path: str | None = None,
) -> Path:
    """Persist real metrics from the current run for the API and README table."""

    output = MODEL_DIR / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    normalized_artifact = artifact_path
    if artifact_path:
        try:
            normalized_artifact = str(Path(artifact_path).resolve().relative_to(ROOT))
        except ValueError:
            normalized_artifact = artifact_path
    payload = {
        "task": task,
        "model": model,
        "dataset": dataset,
        "split": split,
        "metrics": metrics,
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_path": normalized_artifact,
    }
    path = output / f"{task}_{model.lower().replace(' ', '_')}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_confusion_matrix(
    task: str, model: str, y_true: Any, y_pred: Any, labels: list[str]
) -> Path:
    output = MODEL_DIR / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    payload = {"labels": labels, "matrix": matrix}
    path = output / f"{task}_{model.lower().replace(' ', '_')}_confusion.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def trainer_metrics(eval_prediction: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return classification_metrics(labels, predictions)


set_seed(SEED)
