from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from support_assistant.matching import SimilarityIndex
from support_assistant.models import Ticket, TicketQuery
from support_assistant.privacy import display_text, redact_text
from support_assistant.repository import load_history


DATA = Path(__file__).resolve().parents[1] / "starter_kit" / "data" / "tickets.json"


@pytest.mark.parametrize("text,sensitive", [
    ("Reply to demo@example.invalid", "demo@example.invalid"),
    ("api_key=DEMO_ONLY_NOT_A_KEY connection failed", "DEMO_ONLY_NOT_A_KEY"),
    ("Authorization: Bearer SYNTHETIC_NOT_A_TOKEN", "SYNTHETIC_NOT_A_TOKEN"),
    ("phone: +1 (202) 555-0123", "555-0123"),
    ("card_number=4111111111111111", "4111111111111111"),
])
def test_limited_redaction(text, sensitive):
    assert sensitive not in redact_text(text)
    assert "REDACTED" in redact_text(text)


def test_redaction_is_idempotent():
    text = redact_text("Email demo@example.invalid api_key=DEMO_ONLY_NOT_A_KEY")
    assert redact_text(text) == text


def test_query_masks_before_retrieval():
    query = TicketQuery("Login failure", "demo@example.invalid password=DEMO_ONLY_NOT_A_PASSWORD")
    assert "demo@example.invalid" not in query.description
    assert "DEMO_ONLY_NOT_A_PASSWORD" not in query.description


def test_markdown_links_are_rendered_as_literal_text():
    escaped = display_text("[reset](https://example.invalid) **urgent** ![tracking](https://example.invalid/pixel)")
    assert "\\[reset\\]" in escaped
    assert "\\!\\[tracking\\]" in escaped
    assert "\\*\\*urgent\\*\\*" in escaped


@pytest.mark.parametrize("state,status", [("deprecated", "resolved"), ("needs_review", "resolved"), ("approved", "open")])
def test_unapproved_resolutions_never_enter_index(state, status):
    ticket = replace(load_history(DATA)[0], resolution_state=state, status=status)
    assert not SimilarityIndex([ticket]).tickets


@pytest.mark.parametrize("reviewed_at,expected", [(None, False), ("2025-10-20", False), ("2026-10-20", True), ("2026-11-20", False)])
def test_freshness_filter(reviewed_at, expected):
    ticket = replace(load_history(DATA)[0], reviewed_at=reviewed_at)
    assert bool(SimilarityIndex([ticket], max_age_days=180, today=date(2026, 10, 31)).tickets) is expected


@pytest.mark.parametrize("change", [{"status": "unknown"}, {"resolution_state": []}, {"reviewed_at": "2025-09-01"}, {"tags": ["a"] * 21}])
def test_invalid_review_metadata(change):
    document = {**load_history(DATA)[0].to_dict(), **change}
    with pytest.raises(ValueError):
        Ticket.from_dict(document)