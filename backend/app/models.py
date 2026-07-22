"""Persistence models."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Ticket(Base):
    """A support ticket and its latest machine-generated analysis."""

    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    subject: Mapped[str | None] = mapped_column(String(240), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    customer_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    channel: Mapped[str] = mapped_column(String(30), default="email", index=True)
    status: Mapped[str] = mapped_column(String(30), default="open", index=True)
    category: Mapped[str] = mapped_column(String(100), index=True)
    intent_confidence: Mapped[float] = mapped_column(Float)
    sentiment: Mapped[str] = mapped_column(String(20), index=True)
    sentiment_confidence: Mapped[float] = mapped_column(Float)
    urgency: Mapped[str] = mapped_column(String(20), index=True)
    urgency_confidence: Mapped[float] = mapped_column(Float)
    summary: Mapped[str] = mapped_column(Text)
    suggested_response: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
