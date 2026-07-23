"""Shared deterministic preprocessing and weak-labeling functions."""

import random
import re
from dataclasses import dataclass

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch when it is installed."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def clean_text(text: str) -> str:
    """Normalize spacing and redact common sensitive patterns."""

    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", text)
    text = re.sub(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)", "[PHONE]", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class WeakUrgencyResult:
    label: str
    score: int
    matched_signals: tuple[str, ...]


URGENCY_SIGNALS: dict[str, tuple[str, ...]] = {
    "critical": (
        "fraud",
        "stolen",
        "hacked",
        "account hacked",
        "security breach",
        "cannot access funds",
        "emergency",
    ),
    "high": (
        "urgent",
        "immediately",
        "charged twice",
        "blocked",
        "declined",
        "missing money",
        "asap",
    ),
    "low": ("when convenient", "general question", "just wondering", "feedback"),
}


def weak_label_urgency(text: str, sentiment: str = "neutral") -> WeakUrgencyResult:
    """Assign a reproducible synthetic urgency label from auditable rules."""

    normalized = clean_text(text).lower()
    matched: list[str] = []
    score = 1
    for signal in URGENCY_SIGNALS["critical"]:
        if signal in normalized:
            matched.append(signal)
            score += 3
    for signal in URGENCY_SIGNALS["high"]:
        if signal in normalized:
            matched.append(signal)
            score += 2
    for signal in URGENCY_SIGNALS["low"]:
        if signal in normalized:
            matched.append(signal)
            score -= 1
    if sentiment == "negative":
        score += 1
        matched.append("negative_sentiment")
    if re.search(r"\b(?:today|now|hours?)\b", normalized):
        score += 1
        matched.append("time_pressure")
    score = max(0, score)
    label = (
        "critical" if score >= 5 else "high" if score >= 3 else "medium" if score >= 1 else "low"
    )
    return WeakUrgencyResult(label, score, tuple(dict.fromkeys(matched)))
