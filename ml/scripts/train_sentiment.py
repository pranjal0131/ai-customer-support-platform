"""Fine-tune DistilBERT on the TweetEval sentiment configuration."""

from __future__ import annotations

import argparse

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ml.scripts.common import MODEL_DIR, trainer_metrics, write_run_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", default="distilbert-base-uncased")
    args = parser.parse_args()
    dataset = (
        DatasetDict(
            {
                "train": load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train[:1500]"),
                "validation": load_dataset(
                    "cardiffnlp/tweet_eval", "sentiment", split="validation[:400]"
                ),
            }
        )
        if args.smoke
        else load_dataset("cardiffnlp/tweet_eval", "sentiment")
    )
    labels = dataset["train"].features["label"].names
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoded = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=128),
        batched=True,
        remove_columns=["text"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=3,
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )
    output = (
        MODEL_DIR
        / "sentiment"
        / ("distilbert-tweeteval-smoke" if args.smoke else "distilbert-tweeteval")
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output),
            num_train_epochs=1 if args.smoke else 3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            seed=42,
            report_to="none",
            dataloader_pin_memory=torch.cuda.is_available(),
        ),
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=trainer_metrics,
    )
    trainer.train()
    raw = trainer.evaluate()
    metrics = {
        key.removeprefix("eval_"): float(value)
        for key, value in raw.items()
        if key in {"eval_accuracy", "eval_macro_precision", "eval_macro_recall", "eval_macro_f1"}
    }
    trainer.save_model(str(output / "best"))
    tokenizer.save_pretrained(str(output / "best"))
    write_run_metrics(
        task="sentiment",
        model="DistilBERT smoke" if args.smoke else "DistilBERT",
        dataset="cardiffnlp/tweet_eval:sentiment",
        split="validation",
        metrics=metrics,
        artifact_path=str(output / "best"),
    )


if __name__ == "__main__":
    main()
