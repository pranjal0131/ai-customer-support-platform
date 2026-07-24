"""Build a cosine-similarity FAISS index of Bitext support examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from ml.scripts.common import MODEL_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=25_000)
    parser.add_argument("--input", type=Path, help="Optional local Bitext CSV")
    args = parser.parse_args()
    limit = 500 if args.smoke else args.limit
    dataset = (
        load_dataset("csv", data_files=str(args.input), split=f"train[:{limit}]")
        if args.input
        else load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            split=f"train[:{limit}]",
        )
    )
    required = {"instruction", "response"}
    if not required.issubset(dataset.column_names):
        raise ValueError(f"Expected {required}; found {dataset.column_names}")
    records = [
        {
            "instruction": row["instruction"],
            "response": row["response"],
            "intent": row.get("intent", row.get("category", "general_support")),
        }
        for row in dataset
    ]
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(
        [row["instruction"] for row in records],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    output = MODEL_DIR / "semantic"
    output.mkdir(parents=True, exist_ok=True)
    encoder.save(str(output / "encoder"))
    faiss.write_index(index, str(output / "support_examples.faiss"))
    (output / "support_examples.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Indexed {index.ntotal} examples in {output}")


if __name__ == "__main__":
    main()
