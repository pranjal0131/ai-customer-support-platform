"""Export reviewable weak labels and train urgency baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.dataset_sources import load_banking77
from ml.preprocessing import weak_label_urgency
from ml.scripts.common import (
    DATA_DIR,
    MODEL_DIR,
    classification_metrics,
    write_confusion_matrix,
    write_run_metrics,
)


def _load_texts(input_path: Path | None, limit: int) -> list[str]:
    if input_path:
        return [
            json.loads(line)["text"] for line in input_path.read_text(encoding="utf-8").splitlines()
        ]
    dataset = load_banking77(split=f"train[:{limit}]")
    return list(dataset["text"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()
    texts = _load_texts(args.input, args.limit)
    labels = [weak_label_urgency(text, "neutral") for text in texts]
    review_path = DATA_DIR / "processed" / "urgency_weak_labels.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["text", "weak_label", "score", "matched_signals", "reviewed_label"]
        )
        writer.writeheader()
        for text, result in zip(texts, labels, strict=True):
            writer.writerow(
                {
                    "text": text,
                    "weak_label": result.label,
                    "score": result.score,
                    "matched_signals": "|".join(result.matched_signals),
                    "reviewed_label": "",
                }
            )
    y = [item.label for item in labels]
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30_000)
    x_train, x_test = vectorizer.fit_transform(x_train_text), vectorizer.transform(x_test_text)
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    best = ("", None, -1.0)
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = classification_metrics(y_test, predictions)
        write_run_metrics(
            task="urgency_weak_labels",
            model=name,
            dataset="PolyAI/banking77 + deterministic weak rules",
            split="held-out weak labels",
            metrics=metrics,
        )
        write_confusion_matrix("urgency", name, y_test, predictions, sorted(set(y)))
        if metrics["macro_f1"] > best[2]:
            best = (name, model, metrics["macro_f1"])
    output = MODEL_DIR / "urgency"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best[1],
            "vectorizer": vectorizer,
            "label_source": "weak/synthetic; not human ground truth",
        },
        output / "best_model.joblib",
    )
    print(f"Saved {best[0]}; review weak labels at {review_path}")


if __name__ == "__main__":
    main()
