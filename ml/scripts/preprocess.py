"""Normalize downloaded datasets into stable task-specific JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from ml.preprocessing import clean_text
from ml.scripts.common import DATA_DIR


def _records(dataset: Dataset | DatasetDict) -> Dataset:
    return dataset["train"] if isinstance(dataset, DatasetDict) else dataset


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def preprocess_banking(dataset: Dataset) -> list[dict[str, Any]]:
    names = dataset.features["label"].names
    return [
        {"text": clean_text(row["text"]), "label": names[row["label"]]}
        for row in dataset
        if row["text"].strip()
    ]


def preprocess_sentiment(dataset: Dataset) -> list[dict[str, Any]]:
    names = dataset.features["label"].names
    return [
        {"text": clean_text(row["text"]), "label": names[row["label"]]}
        for row in dataset
        if row["text"].strip()
    ]


def preprocess_samsum(dataset: Dataset) -> list[dict[str, Any]]:
    return [
        {"dialogue": clean_text(row["dialogue"]), "summary": clean_text(row["summary"])}
        for row in dataset
        if row["dialogue"].strip() and row["summary"].strip()
    ]


def preprocess_bitext(dataset: Dataset) -> list[dict[str, Any]]:
    return [
        {
            "instruction": clean_text(row["instruction"]),
            "response": clean_text(row["response"]),
            "intent": row.get("intent", row.get("category", "general_support")),
        }
        for row in dataset
        if row["instruction"].strip() and row["response"].strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["all", "intent", "sentiment", "summary", "response"], default="all"
    )
    args = parser.parse_args()
    raw, processed = DATA_DIR / "raw", DATA_DIR / "processed"
    jobs = {
        "intent": ("banking77", preprocess_banking),
        "sentiment": ("tweet_eval_sentiment", preprocess_sentiment),
        "summary": ("samsum", preprocess_samsum),
        "response": ("bitext_support", preprocess_bitext),
    }
    selected = jobs.items() if args.task == "all" else [(args.task, jobs[args.task])]
    for task, (folder, transform) in selected:
        dataset = _records(load_from_disk(str(raw / folder)))
        records = transform(dataset)
        output = processed / f"{task}.jsonl"
        _write_jsonl(output, records)
        print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
