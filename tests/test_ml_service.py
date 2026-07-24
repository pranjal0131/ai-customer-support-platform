"""Tests for inference fallbacks available without downloaded model weights."""

from backend.app.ml_service import ModelService


def test_demo_predictions_are_bounded_and_explainable(tmp_path) -> None:
    service = ModelService(tmp_path)
    intent = service.predict_intent("The ATM declined my cash withdrawal")
    sentiment = service.predict_sentiment("This is awful and my cash is missing")
    urgency = service.predict_urgency("My account was hacked and I need help now", "negative")
    assert intent.label == "cash_withdrawal"
    assert sentiment.label == "negative"
    assert urgency.label == "critical"
    assert all(0 <= item.confidence <= 1 for item in (intent, sentiment, urgency))


def test_summary_is_concise_and_redacted(tmp_path) -> None:
    service = ModelService(tmp_path)
    text = "My email is person@example.com. The card payment failed. " + "Details " * 100
    summary = service.summarize(text)
    assert len(summary) <= 220
    assert "person@example.com" not in summary


def test_similarity_orders_related_text_higher(tmp_path) -> None:
    service = ModelService(tmp_path)
    related = service.token_similarity("card payment was declined", "my card payment failed")
    unrelated = service.token_similarity("card payment was declined", "where is my parcel delivery")
    assert related > unrelated
