"""Idempotent demo-data seeding used for local first-run experience."""

import json

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.ml_service import ModelService
from backend.app.models import Ticket


def seed_if_empty(db: Session, service: ModelService) -> int:
    """Seed reviewed sample tickets only when the database is empty."""

    if (db.scalar(select(func.count(Ticket.id))) or 0) > 0:
        return 0
    path = settings.data_dir / "demo_tickets.json"
    if not path.exists():
        return 0
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        text = record["text"]
        intent = service.predict_intent(text)
        sentiment = service.predict_sentiment(text)
        urgency = service.predict_urgency(text, sentiment.label)
        db.add(
            Ticket(
                subject=record.get("subject"),
                text=text,
                customer_name=record.get("customer_name"),
                channel=record.get("channel", "email"),
                status=record.get("status", "open"),
                category=intent.label,
                intent_confidence=intent.confidence,
                sentiment=sentiment.label,
                sentiment_confidence=sentiment.confidence,
                urgency=urgency.label,
                urgency_confidence=urgency.confidence,
                summary=service.summarize(text),
                suggested_response=service.suggest_response(intent.label, sentiment.label),
            )
        )
    db.commit()
    return len(records)
