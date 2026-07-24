"""Fine-tune DistilBERT for Banking77 intent classification."""

from __future__ import annotations

import argparse
import json

import torch
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ml.dataset_sources import load_banking77
from ml.scripts.common import MODEL_DIR, trainer_metrics, write_run_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--checkpoint", default="distilbert-base-uncased")
    args = parser.parse_args()
    splits = (
        DatasetDict(
            {
                "train": load_banking77(split="train[:1200]"),
                "test": load_banking77(split="test[:400]"),
            }
        )
        if args.smoke
        else load_banking77()
    )
    labels = splits["train"].features["label"].names
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoded = splits.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=192),
        batched=True,
        remove_columns=["text"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )
    output = MODEL_DIR / "intent" / ("distilbert-smoke" if args.smoke else "distilbert")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output),
            num_train_epochs=1 if args.smoke else args.epochs,
            per_device_train_batch_size=8 if args.smoke else 16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            seed=42,
            report_to="none",
            dataloader_pin_memory=torch.cuda.is_available(),
        ),
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=trainer_metrics,
    )
    trainer.train()
    raw_metrics = trainer.evaluate()
    metrics = {
        key.removeprefix("eval_"): float(value)
        for key, value in raw_metrics.items()
        if key.startswith("eval_")
        and key
        not in {"eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"}
    }
    trainer.save_model(str(output / "best"))
    tokenizer.save_pretrained(str(output / "best"))
    write_run_metrics(
        task="intent",
        model="DistilBERT smoke" if args.smoke else "DistilBERT",
        dataset="PolyAI/banking77",
        split="test" if not args.smoke else "test[:400]",
        metrics=metrics,
        artifact_path=str(output / "best"),
    )
    if args.smoke:
        print(
            "Smoke artifact and metrics saved separately; production model selection was unchanged."
        )
        return
    baseline_path = MODEL_DIR / "intent" / "comparison.json"
    baseline_results = (
        json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else []
    )
    best_baseline = max(
        baseline_results,
        key=lambda item: item.get("macro_f1", -1),
        default={"model": "none", "macro_f1": -1},
    )
    transformer_f1 = metrics.get("macro_f1", -1)
    selection = (
        {
            "kind": "transformer",
            "model": "DistilBERT",
            "macro_f1": transformer_f1,
            "artifact": str((output / "best").relative_to(MODEL_DIR.parent)),
        }
        if transformer_f1 > best_baseline.get("macro_f1", -1)
        else {
            "kind": "sklearn",
            "model": best_baseline["model"],
            "macro_f1": best_baseline["macro_f1"],
            "artifact": str(
                (MODEL_DIR / "intent" / "best_model.joblib").relative_to(MODEL_DIR.parent)
            ),
        }
    )
    (MODEL_DIR / "intent" / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
