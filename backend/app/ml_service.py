"""Artifact-aware inference with transparent deterministic demo fallbacks."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from backend.app.config import settings
from ml.preprocessing import clean_text, weak_label_urgency


@dataclass(frozen=True)
class LabelScore:
    label: str
    confidence: float


INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cash_withdrawal": ("atm", "cash", "withdraw", "cash machine"),
    "card_payment": ("card", "merchant", "purchase", "charged", "payment"),
    "card_arrival": ("card arrive", "card delivery", "new card", "track card"),
    "cash_transfer": ("transfer", "recipient", "beneficiary", "wire"),
    "refund": ("refund", "money back", "reversal", "reimburse"),
    "account_access": ("login", "password", "locked", "access", "verification"),
    "fraud_or_security": ("fraud", "stolen", "hacked", "not mine", "unauthorized"),
    "cash_deposit": ("deposit", "cash deposit", "top up", "cash in"),
    "fees_and_exchange": ("fee", "exchange", "rate", "commission", "currency"),
}

POSITIVE = {"thanks", "thank", "great", "helpful", "love", "resolved", "excellent", "happy"}
NEGATIVE = {
    "angry",
    "awful",
    "bad",
    "broken",
    "declined",
    "disappointed",
    "failed",
    "fraud",
    "missing",
    "stolen",
    "terrible",
    "urgent",
    "wrong",
}


class ModelService:
    """Load persisted models once and expose a stable inference interface."""

    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or settings.model_dir
        self.intent_artifact: dict[str, Any] | None = self._load_joblib("intent/best_model.joblib")
        self.urgency_artifact: dict[str, Any] | None = self._load_joblib(
            "urgency/best_model.joblib"
        )
        self.intent_transformer: Any | None = None
        self.sentiment_transformer = self._load_transformer_pipeline(
            "sentiment/distilbert-tweeteval/best", "text-classification"
        )
        self.summarization_transformer = self._load_transformer_pipeline(
            "summarization/flan-t5-small/best", "summarization"
        )
        selection = self._load_json("intent/selection.json")
        if selection and selection.get("kind") == "transformer":
            self.intent_transformer = self._load_transformer_pipeline(
                "intent/distilbert/best", "text-classification"
            )
        self.semantic_encoder: Any | None = None
        self.semantic_index: Any | None = None
        self.semantic_records: list[dict[str, Any]] = []
        self._load_semantic_index()
        self.demo_mode = self._uses_demo_fallbacks
        if self.demo_mode and not settings.demo_mode:
            raise RuntimeError(
                "Trained artifacts are incomplete and DEMO_MODE is disabled. "
                "Train or mount all core model artifacts."
            )

    @property
    def _uses_demo_fallbacks(self) -> bool:
        return (
            (self.intent_transformer is None and self.intent_artifact is None)
            or self.sentiment_transformer is None
            or self.urgency_artifact is None
            or self.summarization_transformer is None
            or self.semantic_index is None
        )

    def _load_json(self, relative_path: str) -> dict[str, Any] | None:
        path = self.model_dir / relative_path
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _load_transformer_pipeline(self, relative_path: str, task: str) -> Any | None:
        """Load local Transformer artifacts without making a network request."""

        path = self.model_dir / relative_path
        if not path.exists():
            return None
        try:
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model_class = (
                AutoModelForSeq2SeqLM
                if task == "summarization"
                else AutoModelForSequenceClassification
            )
            model = model_class.from_pretrained(path, local_files_only=True)
            return pipeline(task, model=model, tokenizer=tokenizer, device=-1)
        except (ImportError, OSError, ValueError):
            return None

    def _load_semantic_index(self) -> None:
        index_path = self.model_dir / "semantic" / "support_examples.faiss"
        records_path = self.model_dir / "semantic" / "support_examples.json"
        encoder_path = self.model_dir / "semantic" / "encoder"
        if not index_path.exists() or not records_path.exists() or not encoder_path.exists():
            return
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self.semantic_index = faiss.read_index(str(index_path))
            self.semantic_records = json.loads(records_path.read_text(encoding="utf-8"))
            self.semantic_encoder = SentenceTransformer(str(encoder_path), local_files_only=True)
        except (ImportError, OSError, ValueError, RuntimeError, json.JSONDecodeError):
            self.semantic_encoder = None
            self.semantic_index = None
            self.semantic_records = []

    def _load_joblib(self, relative_path: str) -> dict[str, Any] | None:
        path = self.model_dir / relative_path
        if not path.exists():
            return None
        try:
            artifact = joblib.load(path)
            return artifact if isinstance(artifact, dict) else None
        except Exception:
            return None

    @property
    def status(self) -> dict[str, str]:
        return {
            "intent": (
                "distilbert"
                if self.intent_transformer
                else "trained-tfidf"
                if self.intent_artifact
                else "demo-heuristic"
            ),
            "sentiment": "distilbert-tweeteval" if self.sentiment_transformer else "demo-lexical",
            "urgency": "trained" if self.urgency_artifact else "demo-weak-rules",
            "summarization": "flan-t5-small"
            if self.summarization_transformer
            else "demo-extractive",
            "semantic_search": "faiss-minilm" if self.semantic_index else "local-cosine",
        }

    def predict_intent(self, text: str) -> LabelScore:
        if self.intent_transformer:
            result = self.intent_transformer(text, truncation=True)[0]
            return LabelScore(str(result["label"]), round(float(result["score"]), 4))
        if self.intent_artifact:
            vectorizer = self.intent_artifact["vectorizer"]
            model = self.intent_artifact["model"]
            encoder = self.intent_artifact.get("label_encoder")
            features = vectorizer.transform([text])
            encoded = model.predict(features)[0]
            label = str(encoder.inverse_transform([encoded])[0]) if encoder else str(encoded)
            confidence = self._classifier_confidence(model, features)
            return LabelScore(label, confidence)

        normalized = text.lower()
        scores = {
            intent: sum(1 for keyword in keywords if keyword in normalized)
            for intent, keywords in INTENT_KEYWORDS.items()
        }
        label, count = max(scores.items(), key=lambda item: item[1])
        if count == 0:
            return LabelScore("general_support", 0.42)
        return LabelScore(label, min(0.55 + count * 0.1, 0.88))

    def predict_sentiment(self, text: str) -> LabelScore:
        if self.sentiment_transformer:
            result = self.sentiment_transformer(text, truncation=True)[0]
            return LabelScore(str(result["label"]).lower(), round(float(result["score"]), 4))
        words = set(re.findall(r"[a-z']+", text.lower()))
        positive = len(words & POSITIVE)
        negative = len(words & NEGATIVE)
        if positive == negative:
            return LabelScore("neutral", 0.58 if positive == 0 else 0.45)
        magnitude = abs(positive - negative)
        return LabelScore(
            "positive" if positive > negative else "negative", min(0.62 + 0.08 * magnitude, 0.94)
        )

    def predict_urgency(self, text: str, sentiment: str) -> LabelScore:
        if self.urgency_artifact:
            features = self.urgency_artifact["vectorizer"].transform([text])
            model = self.urgency_artifact["model"]
            label = str(model.predict(features)[0])
            return LabelScore(label, self._classifier_confidence(model, features))
        result = weak_label_urgency(text, sentiment)
        confidence = min(0.52 + 0.07 * len(result.matched_signals), 0.87)
        return LabelScore(result.label, confidence)

    @staticmethod
    def _classifier_confidence(model: Any, features: Any) -> float:
        if hasattr(model, "predict_proba"):
            return round(float(model.predict_proba(features).max()), 4)
        if hasattr(model, "decision_function"):
            margin = float(abs(model.decision_function(features)).max())
            return round(1 / (1 + math.exp(-margin)), 4)
        return 0.5

    def summarize(self, text: str, limit: int = 220) -> str:
        cleaned = clean_text(text)
        if self.summarization_transformer:
            result = self.summarization_transformer(
                cleaned, max_new_tokens=80, min_length=8, do_sample=False
            )[0]
            return str(result["summary_text"]).strip()
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        summary = " ".join(sentences[:2])
        if len(summary) <= limit:
            return summary
        clipped = summary[: limit - 1].rsplit(" ", 1)[0]
        return f"{clipped}…"

    @staticmethod
    def token_similarity(left: str, right: str) -> float:
        """Cosine similarity over log-scaled term frequencies for demo search."""

        def tokenize(value: str) -> list[str]:
            return re.findall(r"[a-z0-9]+", value.lower())

        left_counts, right_counts = Counter(tokenize(left)), Counter(tokenize(right))
        vocabulary = left_counts.keys() | right_counts.keys()
        numerator = sum(left_counts[token] * right_counts[token] for token in vocabulary)
        left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
        right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
        return round(numerator / (left_norm * right_norm), 4) if left_norm and right_norm else 0.0

    @staticmethod
    def suggest_response(
        intent: str, sentiment: str, similar: list[dict[str, Any]] | None = None
    ) -> str:
        empathy = (
            "I’m sorry for the trouble this has caused. "
            if sentiment == "negative"
            else "Thank you for getting in touch. "
        )
        topic = intent.replace("_", " ")
        evidence = ""
        if similar:
            evidence = (
                " I’ll review the relevant account details and the resolution used for "
                "similar cases."
            )
        return (
            f"{empathy}I understand your question is about {topic}.{evidence} "
            "I’m reviewing the details now and will confirm the next steps shortly. "
            "For your security, please do not send passwords, PINs, or full card numbers."
        )

    def retrieve_examples(self, text: str, limit: int = 3) -> list[dict[str, Any]]:
        """Retrieve response examples from the optional MiniLM/FAISS artifact."""

        if self.semantic_encoder is None or self.semantic_index is None:
            return []
        embedding = self.semantic_encoder.encode([text], normalize_embeddings=True)
        scores, indexes = self.semantic_index.search(embedding, limit)
        return [
            {**self.semantic_records[index], "similarity": round(float(score), 4)}
            for score, index in zip(scores[0], indexes[0], strict=True)
            if 0 <= index < len(self.semantic_records)
        ]

    def load_metrics(self) -> list[dict[str, Any]]:
        """Read only metrics emitted by completed training runs."""

        metrics_dir = self.model_dir / "metrics"
        if not metrics_dir.exists():
            return []
        runs: list[dict[str, Any]] = []
        for path in sorted(metrics_dir.glob("*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    runs.extend(
                        item
                        for item in value
                        if isinstance(item, dict) and {"task", "model", "metrics"}.issubset(item)
                    )
                elif isinstance(value, dict) and {"task", "model", "metrics"}.issubset(value):
                    runs.append(value)
            except (OSError, json.JSONDecodeError):
                continue
        return runs
