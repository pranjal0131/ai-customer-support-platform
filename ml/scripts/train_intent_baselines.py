"""Train and compare three TF-IDF intent-classification baselines."""

from __future__ import annotations

import argparse
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from ml.dataset_sources import load_banking77
from ml.scripts.common import (
    MODEL_DIR,
    classification_metrics,
    write_confusion_matrix,
    write_run_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Use small deterministic subsets")
    args = parser.parse_args()
    train_slice = "train[:1500]" if args.smoke else "train"
    test_slice = "test[:500]" if args.smoke else "test"
    train = load_banking77(split=train_slice)
    test = load_banking77(split=test_slice)
    label_names = train.features["label"].names
    encoder = LabelEncoder().fit(label_names)
    y_train = encoder.transform([label_names[value] for value in train["label"]])
    y_test = encoder.transform([label_names[value] for value in test["label"]])
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=60_000, sublinear_tf=True
    )
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=42
        ),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.25),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
    }
    results = []
    best_name, best_model, best_f1 = "", None, -1.0
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = classification_metrics(y_test, predictions)
        results.append({"model": name, **metrics})
        write_run_metrics(
            task="intent", model=name, dataset="PolyAI/banking77", split=test_slice, metrics=metrics
        )
        write_confusion_matrix("intent", name, y_test, predictions, list(encoder.classes_))
        if metrics["macro_f1"] > best_f1:
            best_name, best_model, best_f1 = name, model, metrics["macro_f1"]
    output = MODEL_DIR / "intent"
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "best_model.joblib"
    joblib.dump(
        {
            "model": best_model,
            "vectorizer": vectorizer,
            "label_encoder": encoder,
            "model_name": best_name,
        },
        artifact,
    )
    (output / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "selection.json").write_text(
        json.dumps(
            {
                "kind": "sklearn",
                "model": best_name,
                "macro_f1": best_f1,
                "artifact": str(artifact.relative_to(MODEL_DIR.parent)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {best_name} (macro F1={best_f1:.4f}) to {artifact}")


if __name__ == "__main__":
    main()
