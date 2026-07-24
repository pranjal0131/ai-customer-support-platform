"""REST API routes for ticket operations and analytics."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.dependencies import get_model_service
from backend.app.ml_service import ModelService
from backend.app.models import Ticket
from backend.app.schemas import (
    AnalysisResponse,
    AnalyticsCount,
    AnalyticsOverview,
    AnalyzeRequest,
    ModelMetricsResponse,
    Prediction,
    SimilarTicket,
    TicketCreate,
    TicketListResponse,
    TicketResponse,
)

router = APIRouter(prefix="/api")
DbSession = Annotated[Session, Depends(get_db)]
Models = Annotated[ModelService, Depends(get_model_service)]


def _similar_tickets(
    db: Session,
    service: ModelService,
    text: str,
    *,
    exclude_id: str | None = None,
    limit: int = 5,
) -> list[SimilarTicket]:
    candidates = db.scalars(select(Ticket).order_by(Ticket.created_at.desc()).limit(500)).all()
    ranked = sorted(
        (
            (service.token_similarity(text, candidate.text), candidate)
            for candidate in candidates
            if candidate.id != exclude_id
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    return [
        SimilarTicket(
            id=ticket.id,
            subject=ticket.subject,
            text=ticket.text,
            category=ticket.category,
            status=ticket.status,
            similarity=score,
        )
        for score, ticket in ranked[:limit]
        if score > 0
    ]


def _analyze(text: str, db: Session, service: ModelService) -> AnalysisResponse:
    intent = service.predict_intent(text)
    sentiment = service.predict_sentiment(text)
    urgency = service.predict_urgency(text, sentiment.label)
    similar = _similar_tickets(db, service, text)
    retrieved_examples = service.retrieve_examples(text)
    retrieval_context = retrieved_examples or [item.model_dump() for item in similar]
    suggested_response = service.suggest_response(intent.label, sentiment.label, retrieval_context)
    return AnalysisResponse(
        category=Prediction(label=intent.label, confidence=intent.confidence),
        sentiment=Prediction(label=sentiment.label, confidence=sentiment.confidence),
        urgency=Prediction(label=urgency.label, confidence=urgency.confidence),
        summary=service.summarize(text),
        suggested_response=suggested_response,
        similar_tickets=similar,
        demo_mode=service.demo_mode,
    )


@router.post("/tickets/analyze", response_model=AnalysisResponse)
def analyze_ticket(payload: AnalyzeRequest, db: DbSession, service: Models) -> AnalysisResponse:
    """Analyze ticket text without persisting it."""

    return _analyze(payload.text, db, service)


@router.post("/tickets", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
def create_ticket(payload: TicketCreate, db: DbSession, service: Models) -> Ticket:
    """Analyze and persist a support ticket."""

    analysis = _analyze(payload.text, db, service)
    ticket = Ticket(
        subject=payload.subject,
        text=payload.text,
        customer_name=payload.customer_name,
        channel=payload.channel,
        category=analysis.category.label,
        intent_confidence=analysis.category.confidence,
        sentiment=analysis.sentiment.label,
        sentiment_confidence=analysis.sentiment.confidence,
        urgency=analysis.urgency.label,
        urgency_confidence=analysis.urgency.confidence,
        summary=analysis.summary,
        suggested_response=analysis.suggested_response,
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


@router.get("/tickets", response_model=TicketListResponse)
def list_tickets(
    db: DbSession,
    search: str | None = Query(default=None, max_length=200),
    category: str | None = Query(default=None, max_length=100),
    sentiment: str | None = Query(default=None, pattern="^(negative|neutral|positive)$"),
    urgency: str | None = Query(default=None, pattern="^(low|medium|high|critical)$"),
    status_filter: str | None = Query(
        default=None, alias="status", pattern="^(open|pending|resolved|closed)$"
    ),
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> TicketListResponse:
    """Return searchable, filterable ticket history."""

    filters = []
    if search:
        pattern = f"%{search.strip()}%"
        filters.append(or_(Ticket.text.ilike(pattern), Ticket.subject.ilike(pattern)))
    if category:
        filters.append(Ticket.category == category)
    if sentiment:
        filters.append(Ticket.sentiment == sentiment)
    if urgency:
        filters.append(Ticket.urgency == urgency)
    if status_filter:
        filters.append(Ticket.status == status_filter)

    query = select(Ticket).where(*filters)
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    items = db.scalars(query.order_by(Ticket.created_at.desc()).offset(offset).limit(limit)).all()
    return TicketListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/tickets/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: str, db: DbSession) -> Ticket:
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.get("/tickets/{ticket_id}/similar", response_model=list[SimilarTicket])
def get_similar_tickets(ticket_id: str, db: DbSession, service: Models) -> list[SimilarTicket]:
    ticket = db.get(Ticket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return _similar_tickets(db, service, ticket.text, exclude_id=ticket.id)


def _group_counts(db: Session, column: object) -> list[AnalyticsCount]:
    rows = db.execute(select(column, func.count(Ticket.id)).group_by(column)).all()
    return [AnalyticsCount(name=str(name), value=count) for name, count in rows]


@router.get("/analytics/overview", response_model=AnalyticsOverview)
def analytics_overview(db: DbSession) -> AnalyticsOverview:
    total = db.scalar(select(func.count(Ticket.id))) or 0
    open_count = db.scalar(select(func.count(Ticket.id)).where(Ticket.status == "open")) or 0
    critical = db.scalar(select(func.count(Ticket.id)).where(Ticket.urgency == "critical")) or 0
    average = db.scalar(select(func.avg(Ticket.intent_confidence))) or 0.0
    return AnalyticsOverview(
        total_tickets=total,
        open_tickets=open_count,
        critical_tickets=critical,
        average_confidence=round(float(average), 4),
        categories=_group_counts(db, Ticket.category),
        sentiments=_group_counts(db, Ticket.sentiment),
        urgencies=_group_counts(db, Ticket.urgency),
    )


@router.get("/models/metrics", response_model=ModelMetricsResponse)
def model_metrics(service: Models) -> ModelMetricsResponse:
    runs = service.load_metrics()
    return ModelMetricsResponse(
        demo_mode=service.demo_mode,
        message=(
            "No executed training metrics found. Run a training command to populate this view."
            if not runs
            else "Metrics loaded from completed local training runs."
        ),
        runs=runs,
    )
