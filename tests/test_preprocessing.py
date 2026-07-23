"""Unit tests for deterministic preprocessing and weak labeling."""

from ml.preprocessing import clean_text, weak_label_urgency


def test_clean_text_redacts_contact_details() -> None:
    result = clean_text(" Email me at user@example.com or call +1 (212) 555-0100. ")
    assert "user@example.com" not in result
    assert "555-0100" not in result
    assert "[EMAIL]" in result
    assert "[PHONE]" in result


def test_critical_weak_label_is_reproducible() -> None:
    text = "My account was hacked and I cannot access funds. This is an emergency now."
    first = weak_label_urgency(text, "negative")
    second = weak_label_urgency(text, "negative")
    assert first == second
    assert first.label == "critical"
    assert "hacked" in first.matched_signals


def test_low_signal_can_reduce_urgency() -> None:
    result = weak_label_urgency("Just wondering about a general question when convenient.")
    assert result.label == "low"
