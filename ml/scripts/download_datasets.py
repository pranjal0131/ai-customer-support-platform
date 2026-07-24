"""Inspect, validate, and optionally cache every configured Hugging Face dataset."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import requests
from datasets import get_dataset_config_names, load_dataset, load_dataset_builder

from ml.dataset_sources import load_banking77
from ml.scripts.common import DATA_DIR


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_id: str
    config: str | None
    required_columns: tuple[str, ...]
    expected_license: str
    use_safe_loader: bool = False
    local_csv_name: str | None = None


DATASETS = (
    DatasetSpec(
        "banking77",
        "PolyAI/banking77",
        "default",
        ("text", "label"),
        "cc-by-4.0",
        use_safe_loader=True,
    ),
    DatasetSpec(
        "tweet_eval_sentiment",
        "cardiffnlp/tweet_eval",
        "sentiment",
        ("text", "label"),
        "cc-by-3.0",
    ),
    DatasetSpec(
        "samsum",
        "knkarthick/samsum",
        None,
        ("dialogue", "summary"),
        "cc-by-nc-nd-4.0",
    ),
    DatasetSpec(
        "bitext_support",
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        None,
        ("instruction", "response"),
        "cdla-sharing-1.0",
        local_csv_name="bitext_support.csv",
    ),
)


def inspect_dataset(spec: DatasetSpec) -> dict[str, object]:
    """Resolve configs and validate declared feature names before downloading."""

    local_csv = DATA_DIR / "raw" / spec.local_csv_name if spec.local_csv_name else None
    dataset = (
        load_banking77("train[:1]")
        if spec.use_safe_loader
        else load_dataset("csv", data_files=str(local_csv), split="train[:1]")
        if local_csv and local_csv.exists()
        else None
    )
    configs = ["default"] if dataset else get_dataset_config_names(spec.dataset_id)
    if spec.config:
        configs = sorted(set(configs) | {spec.config})
    config = spec.config or (configs[0] if len(configs) == 1 else None)
    builder = None if dataset else load_dataset_builder(spec.dataset_id, config)
    features = dataset.features if dataset else builder.info.features
    columns = list(features.keys())
    missing = sorted(set(spec.required_columns) - set(columns))
    if missing:
        raise ValueError(
            f"{spec.dataset_id} columns changed; missing {missing}. Available columns: {columns}"
        )
    return {
        "key": spec.key,
        "dataset_id": spec.dataset_id,
        "selected_config": config,
        "available_configs": configs,
        "columns": columns,
        "license": _metadata_license(
            spec.dataset_id,
            builder.info.license if builder else None,
            spec.expected_license,
        ),
        "data_only_loader": "local/publisher CSV" if dataset else "native",
        "description": (builder.info.description or "").strip()[:500] if builder else "",
    }


def _metadata_license(dataset_id: str, builder_license: str | None, expected_license: str) -> str:
    """Read Hub metadata with a bounded fallback to the reviewed registry."""

    if os.getenv("HF_HUB_OFFLINE") == "1":
        return str(builder_license or expected_license)
    try:
        response = requests.get(f"https://huggingface.co/api/datasets/{dataset_id}", timeout=10)
        response.raise_for_status()
        card_license = (response.json().get("cardData") or {}).get("license")
    except (requests.RequestException, ValueError):
        card_license = None
    return str(card_license or builder_license or expected_license)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Cache at most 250 rows per split")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--dataset", choices=[spec.key for spec in DATASETS], help="Process one dataset only"
    )
    args = parser.parse_args()
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    selected_specs = [spec for spec in DATASETS if not args.dataset or spec.key == args.dataset]
    for spec in selected_specs:
        manifest = inspect_dataset(spec)
        manifests.append(manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        if args.inspect_only:
            continue
        split = "train[:250]" if args.smoke else None
        dataset = (
            load_banking77(split=split)
            if spec.use_safe_loader
            else load_dataset(
                "csv", data_files=str(DATA_DIR / "raw" / spec.local_csv_name), split=split
            )
            if spec.local_csv_name and (DATA_DIR / "raw" / spec.local_csv_name).exists()
            else load_dataset(spec.dataset_id, manifest["selected_config"], split=split)
        )
        dataset.save_to_disk(str(raw_dir / spec.key))
    Path(raw_dir / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
