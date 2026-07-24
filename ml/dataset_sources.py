"""Data-only Hugging Face source definitions used by training scripts."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, load_dataset

BANKING77_ID = "PolyAI/banking77"
BANKING77_CSV: dict[str, str] = {
    "train": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
        "master/banking_data/train.csv"
    ),
    "test": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
        "master/banking_data/test.csv"
    ),
}
_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_BANKING77 = {
    "train": _ROOT / "data" / "raw" / "banking77_train.csv",
    "test": _ROOT / "data" / "raw" / "banking77_test.csv",
}
_SPLIT_PATTERN = re.compile(r"^(train|test)(?:\[(\d*):(\d*)\])?$")


@lru_cache(maxsize=1)
def _banking77_dataset() -> DatasetDict:
    """Load and type the publisher CSVs with no repository code execution."""

    local_available = all(path.exists() for path in _LOCAL_BANKING77.values())
    data_files = (
        {name: str(path) for name, path in _LOCAL_BANKING77.items()}
        if local_available
        else BANKING77_CSV
    )
    raw = load_dataset("csv", data_files=data_files)
    label_names = sorted(set(raw["train"]["category"]) | set(raw["test"]["category"]))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    features = Features({"text": Value("string"), "label": ClassLabel(names=label_names)})

    def encode(row: dict[str, str]) -> dict[str, str | int]:
        return {"text": row["text"], "label": label_to_id[row["category"]]}

    return DatasetDict(
        {
            name: dataset.map(
                encode,
                remove_columns=dataset.column_names,
                features=features,
                desc=f"Typing Banking77 {name}",
            )
            for name, dataset in raw.items()
        }
    )


def load_banking77(split: str | None = None, **_: Any) -> Dataset | DatasetDict:
    """Load Banking77 through safe publisher CSVs and support common split slices."""

    dataset = _banking77_dataset()
    if split is None:
        return dataset
    match = _SPLIT_PATTERN.fullmatch(split)
    if not match:
        raise ValueError(f"Unsupported Banking77 split expression: {split}")
    name, start_value, end_value = match.groups()
    selected = dataset[name]
    if start_value is None and end_value is None:
        return selected
    start = int(start_value) if start_value else 0
    end = int(end_value) if end_value else len(selected)
    return selected.select(range(start, min(end, len(selected))))
