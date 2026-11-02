import json
from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np

from support_assistant.matching import SimilarityIndex, similarity_level
from support_assistant.models import Ticket, TicketQuery


DATA = Path(__file__).resolve().parents[1] / "starter_kit" / "data"


@pytest.fixture
def index():
    documents = json.loads((DATA / "tickets.json").read_text(encoding="utf-8"))["tickets"]
    return SimilarityIndex(Ticket.from_dict(document) for document in documents)


@pytest.mark.parametrize("ticket_id,expected", [("NT-2", "T-001201"), ("NT-3", "T-001303")])
def test_supplied_positive_examples(index, ticket_id, expected):
    incoming = json.loads((DATA / "new_tickets_example.json").read_text(encoding="utf-8"))["new_tickets"]
    document = next(ticket for ticket in incoming if ticket["ticket_id"] == ticket_id)
    results = index.search(TicketQuery(document["subject"], document["description"]))
    assert results[0].ticket.ticket_id == expected
    assert results[0].matched_terms


def test_snowflake_has_no_strong_precedent(index):
    incoming = json.loads((DATA / "new_tickets_example.json").read_text(encoding="utf-8"))["new_tickets"]
    document = next(ticket for ticket in incoming if ticket["ticket_id"] == "NT-4")
    assert index.search(TicketQuery(document["subject"], document["description"])) == []


@pytest.mark.parametrize("subject,description", [
    ("quasar zebralith", "extraterrestrial telemetry"),
    ("Cafeteria lunch", "Please add vegan meals to the office menu."),
])
def test_unrelated_queries_abstain(index, subject, description):
    assert index.search(TicketQuery(subject, description)) == []


@pytest.mark.parametrize("subject,expected", [
    ("My card has been charged twice this month", "T-001303"),
    ("Revenue chart has no bars or axis labels", "T-001201"),
    ("Okta SAML login returns a blank page", "T-001002"),
])
def test_additional_paraphrases(index, subject, expected):
    results = index.search(TicketQuery(subject, subject))
    assert results[0].ticket.ticket_id == expected


@pytest.mark.parametrize("top_n", [-1, 0, 11, True, 1.5])
def test_invalid_result_limits(index, top_n):
    with pytest.raises(ValueError):
        index.search(TicketQuery("billing charge", "duplicate charge"), top_n=top_n)


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan")])
def test_invalid_threshold(index, score):
    with pytest.raises(ValueError):
        index.search(TicketQuery("billing charge", "duplicate charge"), min_score=score)


@pytest.mark.parametrize("subject,description", [("", "hello"), ("hello", " "), (None, "hello"), ("a" * 301, "hello")])
def test_invalid_queries(subject, description):
    with pytest.raises(ValueError):
        TicketQuery(subject, description)


def test_empty_history():
    assert SimilarityIndex([]).search(TicketQuery("hello", "billing")) == []


def test_zero_overlap_excluded_even_with_zero_threshold(index):
    assert index.search(TicketQuery("quasar", "zebralith"), min_score=0) == []


def test_ambiguous_lexical_query_abstains(index):
    text = "My invoice bills me twice for one payment"
    assert index.search(TicketQuery(text, text)) == []


def test_deterministic_results(index):
    query = TicketQuery("login password", "login password reset")
    assert index.search(query) == index.search(query)


def test_category_filter_runs_before_result_limit(index):
    query = TicketQuery("Dashboard chart", "Dashboard performance chart rendering and billing invoice")
    all_results = index.search(query, top_n=10, min_score=0)
    billing = index.search(query, top_n=1, min_score=0, categories=frozenset({"billing"}))
    assert all_results[0].ticket.category != "billing"
    assert len(billing) == 1
    assert billing[0].ticket.category == "billing"


def test_unknown_category_has_no_matches(index):
    assert index.search(TicketQuery("billing", "duplicate charge"), categories=frozenset({"nonexistent"})) == []


def test_none_category_preserves_existing_rankings(index):
    query = TicketQuery("Billing charge", "My card was charged twice this month")
    assert index.search(query) == index.search(query, categories=None)


def test_assessment_exposes_closest_score_without_recommending_weak_match(index):
    incoming = json.loads((DATA / "new_tickets_example.json").read_text(encoding="utf-8"))["new_tickets"]
    document = next(ticket for ticket in incoming if ticket["ticket_id"] == "NT-4")
    assessment = index.assess(TicketQuery(document["subject"], document["description"]))
    assert not assessment.matches
    assert 0 < assessment.closest_score < 0.30
    assert assessment.candidate_count == 40


def test_closest_score_respects_category_filter(index):
    query = TicketQuery("Billing charge", "My card was charged twice this month")
    filtered = index.assess(query, min_score=0, categories=frozenset({"auth"}))
    assert all(match.ticket.category == "auth" for match in filtered.matches)
    assert filtered.closest_score < index.assess(query).closest_score
    assert filtered.candidate_count == 5


def test_three_strong_matches_are_ranked_without_padding(index):
    original = index.tickets[0]
    candidates = [replace(original, ticket_id=f"T-{900000 + number:06d}", legacy_ids=()) for number in range(4)]
    assessment = SimilarityIndex(candidates).assess(TicketQuery(original.subject, original.description))
    assert len(assessment.matches) == 3
    assert [match.ticket.ticket_id for match in assessment.matches] == ["T-900000", "T-900001", "T-900002"]
    assert all(match.score >= 0.30 for match in assessment.matches)


def test_no_candidates_does_not_fabricate_a_closest_score(index):
    assessment = index.assess(TicketQuery("Chart", "Empty chart"), categories=frozenset({"unknown"}))
    assert assessment.closest_score is None
    assert assessment.candidate_count == 0


def test_semantic_similarity_can_match_without_shared_terms(index):
    class Encoder:
        def encode(self, texts, **kwargs):
            return np.asarray([
                [1.0, 0.0] if "quasarquery" in text else
                [0.98, 0.2] if "semantichit" in text else
                [0.0, 1.0]
                for text in texts
            ])

    candidates = [
        replace(index.tickets[0], ticket_id="T-900101", legacy_ids=(), subject="semantichit", description="unrelated vocabulary", tags=()),
        replace(index.tickets[1], ticket_id="T-900102", legacy_ids=(), subject="semanticmiss", description="different vocabulary", tags=()),
    ]
    semantic_index = SimilarityIndex(candidates, semantic_encoder=Encoder())
    results = semantic_index.search(TicketQuery("quasarquery", "unknown phrasing"))
    assert [match.ticket.ticket_id for match in results] == ["T-900101"]
    assert results[0].matched_terms == ()
    assert results[0].semantic_score > 0.9


def test_missing_semantic_model_warns_and_preserves_lexical_search(index, monkeypatch):
    monkeypatch.setattr("support_assistant.matching.semantic_model_source", lambda: None)
    fallback = SimilarityIndex(index.tickets[:2])
    query = TicketQuery(index.tickets[0].subject, index.tickets[0].description)
    assert fallback.semantic_matrix is None
    assert "Falling back to TF-IDF and tag matching" in fallback.semantic_warning
    assert fallback.search(query)[0].ticket.ticket_id == index.tickets[0].ticket_id


def test_loaded_ticket_is_excluded_from_its_own_results(index):
    ticket = index.tickets[0]
    query = TicketQuery(ticket.subject, ticket.description)
    assert index.assess(query).matches[0].ticket.ticket_id == ticket.ticket_id
    assessment = index.assess(query, min_score=0, exclude_ticket_id=ticket.ticket_id)
    assert all(match.ticket.ticket_id != ticket.ticket_id for match in assessment.matches)
    assert assessment.candidate_count == len(index.tickets) - 1


@pytest.mark.parametrize("score,expected", [(0.9, "High similarity"), (0.8, "High similarity"),
                                          (0.5, "Moderate similarity"), (0.49, "Limited similarity")])
def test_similarity_bands_are_not_probabilities(score, expected):
    assert similarity_level(score) == expected

def test_review_age_is_applied_per_query_without_rebuilding():
    from datetime import date

    reviewed = Ticket.from_dict({"ticket_id": "T-009301", "category": "api", "subject": "Webhook signature rejected",
                                 "description": "Receiver rejects webhook signatures after rotation.", "resolution_notes": "Rotate the secret.",
                                 "tags": [], "created_at": "2026-01-01", "status": "resolved", "reviewed_at": "2026-01-10"})
    undated = replace(reviewed, ticket_id="T-009302", legacy_ids=(), reviewed_at=None)
    index = SimilarityIndex([reviewed, undated])
    query = TicketQuery("Webhook signature rejected", "Receiver rejects webhook signatures after rotation.")
    assert {match.ticket.ticket_id for match in index.search(query)} == {"T-009301", "T-009302"}
    fresh = index.search(query, max_age_days=180, today=date(2026, 3, 1))
    assert [match.ticket.ticket_id for match in fresh] == ["T-009301"]
    assert index.search(query, max_age_days=180, today=date(2026, 12, 1)) == []
    assert int(index.eligible_mask(180, date(2026, 3, 1)).sum()) == 1
    assert len(index.indexed) == 2  # both stay indexed; only eligibility changed
