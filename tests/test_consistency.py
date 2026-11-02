import numpy as np
import pytest

from support_assistant.config import ROOT
from support_assistant.consistency import conflict_report, conflicts_among
from support_assistant.matching import SimilarityIndex
from support_assistant.models import Ticket
from support_assistant.repository import load_history
from support_assistant.semantic import semantic_model_source


def ticket(ticket_id, subject, resolution):
    return Ticket.from_dict({"ticket_id": ticket_id, "category": "billing", "subject": subject,
                             "description": subject + " reported by the customer.", "resolution_notes": resolution,
                             "tags": [], "created_at": "2026-01-01", "status": "resolved"})


class MarkerEncoder:
    """Embeds by marker word so issue similarity is explicit: same marker = near-duplicate issue."""

    def encode(self, texts, **kwargs):
        vectors = {"doublecharge": [1.0, 0.0, 0.0], "lookalike": [0.7, 0.7, 0.0], "unrelated": [0.0, 0.0, 1.0]}
        return np.asarray([next(vector for marker, vector in vectors.items() if marker in text) for text in texts])


REFUND = "Issue a refund for the duplicate settled charge and confirm the reversal with the payment processor."
WAIT = "Explain that the pending authorization hold expires automatically within seven days; no refund is needed."
REFUND_REWORDED = "Refund the duplicate settled charge and confirm the reversal with the payment processor team."


@pytest.fixture
def tickets():
    return [
        ticket("T-009201", "doublecharge billed twice", REFUND),
        ticket("T-009202", "doublecharge charged two times", WAIT),
        ticket("T-009203", "doublecharge duplicate payment", REFUND_REWORDED),
        ticket("T-009204", "lookalike pending hold", WAIT),
        ticket("T-009205", "unrelated export timeout", "Split the export into smaller date ranges."),
    ]


def test_similar_precedents_with_different_resolutions_are_flagged(tickets):
    index = SimilarityIndex(tickets, semantic_encoder=MarkerEncoder())
    flagged = {(conflict.first.ticket_id, conflict.second.ticket_id) for conflict in conflicts_among(index, tickets)}
    assert ("T-009201", "T-009202") in flagged  # same issue, refund vs wait
    assert ("T-009201", "T-009204") in flagged  # look-alike issue (cosine 0.71), different fix
    assert ("T-009201", "T-009203") not in flagged  # same fix in different words
    assert not any("T-009205" in pair for pair in flagged)  # unrelated issue is never compared as a conflict


def test_report_lists_only_near_duplicate_issues_answered_differently(tickets):
    index = SimilarityIndex(tickets, semantic_encoder=MarkerEncoder())
    reported = {(conflict.first.ticket_id, conflict.second.ticket_id) for conflict in conflict_report(index)}
    # The 0.71 look-alike warns during search but is not reported as an inconsistent answer (threshold 0.75).
    assert reported == {("T-009201", "T-009202"), ("T-009202", "T-009203")}


def test_lexical_fallback_uses_its_own_scale(tickets):
    index = SimilarityIndex(tickets)
    assert index.semantic_matrix is None
    flagged = {(conflict.first.ticket_id, conflict.second.ticket_id) for conflict in conflicts_among(index, tickets)}
    assert not any("T-009205" in pair for pair in flagged)
    assert ("T-009201", "T-009203") not in flagged


@pytest.mark.semantic
@pytest.mark.skipif(semantic_model_source() is None, reason="Download the model with: python -m support_assistant.semantic")
def test_real_history_flags_duplicate_charge_versus_pending_authorization():
    history = load_history(ROOT / "starter_kit" / "data" / "tickets.json") + load_history(ROOT / "data" / "production_tickets.json")
    index = SimilarityIndex(history)
    by_id = {ticket.ticket_id: ticket for ticket in history}
    conflicts = conflicts_among(index, [by_id["T-001303"], by_id["T-002010"]])
    assert [(conflict.first.ticket_id, conflict.second.ticket_id) for conflict in conflicts] == [("T-001303", "T-002010")]
    assert conflict_report(index) == ()  # the synthetic history has no same-issue, different-answer pairs
