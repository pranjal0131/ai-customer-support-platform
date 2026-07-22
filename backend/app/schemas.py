"""Typed API request and response contracts."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Channel = Literal["email", "chat", "phone", "social", "web"]
TicketStatus = Literal["open", "pending", "resolved", "closed"]
Urgency = Literal["low", "medium", "high", "critical"]
Sentiment = Literal["negative", "neutral", "positive"]


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=3, max_length=10_000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Ticket text cannot be blank")
        return value


class TicketCreate(AnalyzeRequest):
    subject: str | None = Field(default=None, max_length=240)
    customer_name: str | None = Field(default=None, max_length=120)
    channel: Channel = "email"


class Prediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class SimilarTicket(BaseModel):
    id: str
    subject: str | None
    text: str
    category: str
    status: str
    similarity: float = Field(ge=0, le=1)


class AnalysisResponse(BaseModel):
    category: Prediction
    sentiment: Prediction
    urgency: Prediction
    summary: str
    suggested_response: str
    similar_tickets: list[SimilarTicket] = Field(default_factory=list)
    ai_suggestion_requires_review: bool = True
    demo_mode: bool


class TicketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    subject: str | None
    text: str
    customer_name: str | None
    channel: str
    status: str
    category: str
    intent_confidence: float
    sentiment: str
    sentiment_confidence: float
    urgency: str
    urgency_confidence: float
    summary: str
    suggested_response: str
    created_at: datetime
    updated_at: datetime


class TicketListResponse(BaseModel):
    items: list[TicketResponse]
    total: int
    limit: int
    offset: int


class AnalyticsCount(BaseModel):
    name: str
    value: int


class AnalyticsOverview(BaseModel):
    total_tickets: int
    open_tickets: int
    critical_tickets: int
    average_confidence: float
    categories: list[AnalyticsCount]
    sentiments: list[AnalyticsCount]
    urgencies: list[AnalyticsCount]


class ModelMetric(BaseModel):
    task: str
    model: str
    dataset: str
    split: str
    metrics: dict[str, float]
    created_at: str | None = None
    artifact_path: str | None = None


class ModelMetricsResponse(BaseModel):
    demo_mode: bool
    message: str
    runs: list[ModelMetric]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    demo_mode: bool
    model_status: dict[str, str]
