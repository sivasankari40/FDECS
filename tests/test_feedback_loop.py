from datetime import date
from pathlib import Path

import pytest

from support_assistant.feedback import FeedbackStore, feedback_report
from support_assistant.matching import SimilarityIndex
from support_assistant.models import Ticket, TicketQuery
from support_assistant.replay import evaluate_feedback, feedback_cases, sweep
from support_assistant.repository import load_history
from support_assistant.config import Settings
from unittest.mock import Mock


DATA = Path(__file__).resolve().parents[1] / "starter_kit" / "data" / "tickets.json"
SEARCH = "a" * 32


@pytest.fixture
def index():
    return SimilarityIndex(load_history(DATA))


def billing_query(codes=()):
    return TicketQuery("Charged twice for the same billing cycle", "Customer was billed twice on their card this month.", codes)


def test_absence_reasons_cover_every_way_a_precedent_is_hidden(index):
    query = billing_query()
    shown = [match.ticket.ticket_id for match in index.search(query, top_n=3, min_score=0)]
    assert index.explain_absence(query, "T-999999") == ("not_indexed", None)
    assert index.explain_absence(query, "T-001303", max_age_days=180, today=date(2026, 10, 31)) == ("outside_review_window", None)
    assert index.explain_absence(query, "T-001303", categories=frozenset({"auth"})) == ("category_filter", None)
    reason, score = index.explain_absence(query, "T-001601")  # dashboard performance: unrelated to billing
    assert reason == "below_threshold" and 0 <= score < 0.30
    reason, score = index.explain_absence(query, shown[2], min_score=0, top_n=2)
    assert reason == "ranked_below_shown" and score > 0
    assert index.explain_absence(query, shown[0], min_score=0)[0] == "shown"


def test_conflicting_error_code_is_reported_as_the_reason():
    ticket = Ticket.from_dict({"ticket_id": "T-009401", "category": "api", "subject": "API request rejected",
                               "description": "Client receives an error.", "resolution_notes": "Fix.", "tags": [],
                               "created_at": "2026-01-01", "status": "resolved", "error_codes": ["HTTP 500"]})
    index = SimilarityIndex([ticket])
    query = TicketQuery("API request rejected", "Client receives an error.", ("HTTP 403",))
    assert index.explain_absence(query, "T-009401") == ("error_code_conflict", None)


def vote(ticket_id, helpful, score, incoming="T-005003", search=SEARCH, voted_at="2026-11-02T10:00:00+00:00"):
    return {"search_id": search, "ticket_id": ticket_id, "incoming_ticket_id": incoming, "helpful": helpful,
            "rank": 1, "score": score, "voted_at": voted_at}


def test_report_shows_helpful_rate_by_score_band_review_list_and_trend():
    votes = [vote("T-001303", True, 0.66), vote("T-001303", True, 0.70, search="b" * 32),
             vote("T-001302", False, 0.33), vote("T-001302", False, 0.35, search="c" * 32), vote("T-001302", True, 0.38, search="d" * 32),
             vote("T-001302", False, 0.36, search="e" * 32, voted_at="2026-11-10T09:00:00+00:00")]
    missed = [{"ticket_id": "T-002010", "reason": "below_threshold", "reported_at": "2026-11-10T09:05:00+00:00"},
              {"ticket_id": "T-002010", "reason": "below_threshold", "reported_at": "2026-11-10T09:06:00+00:00"},
              {"ticket_id": "T-001201", "reason": "category_filter", "reported_at": "2026-11-10T09:07:00+00:00"}]
    report = feedback_report(votes, missed)
    bands = {band["score_band"]: band for band in report["by_score_band"]}
    assert bands["0.00-0.40"] == {"score_band": "0.00-0.40", "votes": 4, "helpful_rate": 0.25}
    assert bands["0.60-0.80"] == {"score_band": "0.60-0.80", "votes": 2, "helpful_rate": 1.0}
    assert report["precedents_to_review"] == [{"ticket_id": "T-001302", "helpful": 1, "not_helpful": 3, "helpful_rate": 0.25}]
    assert report["missed_precedents"]["by_reason"] == {"below_threshold": 2, "category_filter": 1}
    assert report["missed_precedents"]["most_reported"][0] == {"ticket_id": "T-002010", "reports": 2}
    assert [week["week"] for week in report["weekly"]] == ["2026-W45", "2026-W46"]
    assert report["weekly"][1] == {"week": "2026-W46", "votes": 1, "helpful_rate": 0.0, "missed_reports": 3}


def test_precedents_need_enough_votes_before_review():
    assert feedback_report([vote("T-001302", False, 0.3)], [])["precedents_to_review"] == []


def test_missed_report_stores_reason_and_shown_ids_only():
    store = FeedbackStore(Settings())
    store.repository._request = Mock(side_effect=[{"error": "not_found"}, {"ok": True}])
    store.record_missed(SEARCH, "t-2010", "T-005003", ("T-001303",), "below_threshold", 0.2412)
    document = store.repository._request.call_args_list[1].kwargs["json"]
    assert document["_id"] == f"missed:{SEARCH}:T-002010"
    assert (document["reason"], document["score"], document["shown_ticket_ids"]) == ("below_threshold", 0.2412, ["T-001303"])
    with pytest.raises(ValueError):
        store.record_missed(SEARCH, "T-002010", None, (), "shown", 0.9)


def test_votes_and_missed_reports_become_labelled_cases(index):
    incoming = {"T-005003": Ticket.incoming("T-005003", "Getting charged twice this month", "Card statement shows two charges for one payment.")}
    votes = [vote("T-001303", True, 0.66), vote("T-001303", False, 0.66, search="b" * 32), vote("T-001303", True, 0.66, search="c" * 32),
             vote("T-001302", False, 0.33),
             vote("T-001101", True, 0.31), vote("T-001101", False, 0.31, search="b" * 32),
             vote("T-001201", True, 0.5, incoming=None), vote("T-001201", True, 0.5, incoming="T-005099")]
    missed = [{"ticket_id": "T-001304", "incoming_ticket_id": "T-005003", "reason": "below_threshold"}]
    cases, skipped = feedback_cases(votes, missed, incoming.get)
    assert len(cases) == 1
    assert cases[0]["relevant"] == ["T-001303", "T-001304"]  # majority helpful + reported as missed
    assert cases[0]["irrelevant"] == ["T-001302"]  # T-001101 is a tie and is dropped
    assert skipped == {"votes_without_incoming_ticket": 1, "incoming_ticket_no_longer_available": 1}


def test_replay_scores_settings_and_keeps_unjudged_results_visible(index):
    case = {"incoming_ticket_id": "T-005003", "subject": "Getting charged twice this month",
            "description": "Customer's card statement shows two charges for a single monthly payment.", "error_codes": [],
            "relevant": ["T-001303"], "irrelevant": ["T-001302"]}
    strict = evaluate_feedback(index, [case], 0.30)
    assert strict["helpful_recall"] == 1.0 and strict["cases_with_a_helpful_result"] == 1
    loose = evaluate_feedback(index, [case], 0.0)
    assert loose["unjudged_results"] >= strict["unjudged_results"]
    rows = sweep(index, [case])
    assert [row["threshold"] for row in rows] == [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    assert index.semantic_weight == 0.45
