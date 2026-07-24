"""Optional lightweight FLAN-T5 fine-tuning on SAMSum."""

from __future__ import annotations

import argparse

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from ml.scripts.common import MODEL_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", default="google/flan-t5-small")
    args = parser.parse_args()
    dataset = (
        DatasetDict(
            {
                "train": load_dataset("knkarthick/samsum", split="train[:500]"),
                "validation": load_dataset("knkarthick/samsum", split="validation[:100]"),
            }
        )
        if args.smoke
        else load_dataset("knkarthick/samsum")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, object]:
        inputs = tokenizer(
            [f"summarize: {text}" for text in batch["dialogue"]], max_length=512, truncation=True
        )
        labels = tokenizer(text_target=batch["summary"], max_length=96, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    encoded = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    output = (
        MODEL_DIR / "summarization" / ("flan-t5-small-smoke" if args.smoke else "flan-t5-small")
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(output),
            num_train_epochs=1 if args.smoke else 3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            learning_rate=3e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            predict_with_generate=True,
            seed=42,
            report_to="none",
            dataloader_pin_memory=torch.cuda.is_available(),
        ),
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(str(output / "best"))
    tokenizer.save_pretrained(str(output / "best"))
    print("Model saved. Run evaluate.py to produce evaluation metrics; none are fabricated here.")


if __name__ == "__main__":
    main()
