"""API contract and workflow tests."""

from fastapi.testclient import TestClient


def test_health_reports_model_status(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "intent" in body["model_status"]


def test_analyze_does_not_persist(client: TestClient) -> None:
    response = client.post(
        "/api/tickets/analyze",
        json={"text": "My card was charged twice and I need a refund urgently."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["category"]["label"] in {"card_payment", "refund"}
    assert body["ai_suggestion_requires_review"] is True
    assert client.get("/api/tickets").json()["total"] == 0


def test_create_list_get_and_similar_workflow(client: TestClient) -> None:
    first = client.post(
        "/api/tickets",
        json={"subject": "ATM issue", "text": "The ATM did not give me cash", "channel": "chat"},
    )
    second = client.post(
        "/api/tickets",
        json={"subject": "Cash machine", "text": "My cash withdrawal at the ATM failed"},
    )
    assert first.status_code == 201
    assert second.status_code == 201
    ticket_id = first.json()["id"]
    assert client.get(f"/api/tickets/{ticket_id}").status_code == 200
    listing = client.get("/api/tickets", params={"search": "ATM", "urgency": "medium"})
    assert listing.status_code == 200
    assert listing.json()["total"] >= 1
    similar = client.get(f"/api/tickets/{ticket_id}/similar")
    assert similar.status_code == 200
    assert similar.json()[0]["id"] == second.json()["id"]


def test_analytics_use_persisted_tickets(client: TestClient) -> None:
    client.post(
        "/api/tickets", json={"text": "Urgent: my account was hacked and funds are missing now"}
    )
    response = client.get("/api/analytics/overview")
    assert response.status_code == 200
    assert response.json()["total_tickets"] == 1
    assert response.json()["critical_tickets"] == 1


def test_validation_and_not_found_errors(client: TestClient) -> None:
    invalid = client.post("/api/tickets/analyze", json={"text": "   "})
    missing = client.get("/api/tickets/not-a-real-id")
    assert invalid.status_code == 422
    assert invalid.json()["error"] == "validation_error"
    assert missing.status_code == 404


def test_metrics_are_empty_until_a_run_exists(client: TestClient) -> None:
    response = client.get("/api/models/metrics")
    assert response.status_code == 200
    assert response.json()["runs"] == []
