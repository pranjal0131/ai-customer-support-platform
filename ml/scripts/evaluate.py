"""Evaluate a saved sklearn classifier on an untouched Hugging Face split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from ml.dataset_sources import load_banking77
from ml.scripts.common import classification_metrics, write_confusion_matrix, write_run_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path("models/intent/best_model.joblib"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    artifact = joblib.load(args.artifact)
    split = "test[:500]" if args.smoke else "test"
    dataset = load_banking77(split=split)
    names = dataset.features["label"].names
    encoder = artifact["label_encoder"]
    y_true = encoder.transform([names[value] for value in dataset["label"]])
    predictions = artifact["model"].predict(artifact["vectorizer"].transform(dataset["text"]))
    metrics = classification_metrics(y_true, predictions)
    model_name = artifact.get("model_name", type(artifact["model"]).__name__)
    write_run_metrics(
        task="intent",
        model=model_name,
        dataset="PolyAI/banking77",
        split=split,
        metrics=metrics,
        artifact_path=str(args.artifact),
    )
    write_confusion_matrix("intent", model_name, y_true, predictions, list(encoder.classes_))
    print(metrics)


if __name__ == "__main__":
    main()
