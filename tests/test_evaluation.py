from datetime import date

import pytest

from support_assistant.config import ROOT
from support_assistant.evaluation import CORPUS_AS_OF, evaluate, load_cases
from support_assistant.matching import SimilarityIndex
from support_assistant.repository import load_history
from support_assistant.semantic import semantic_model_source


def test_synthetic_corpus_integrity_and_review_states():
    history = load_history(ROOT / "data" / "production_tickets.json")
    assert len(history) == 24
    assert len({ticket.ticket_id for ticket in history}) == 24
    index = SimilarityIndex(history, max_age_days=180, today=date(2026, 10, 31))
    excluded = {"T-002007", "T-002018", "T-002021", "T-002022", "T-002023", "T-002024"}
    assert excluded.isdisjoint(ticket.ticket_id for ticket in index.tickets)


def test_evaluation_split_integrity():
    cases = load_cases()
    assert len(cases) == 32
    development = {case["case_id"] for case in cases if case["split"] == "development"}
    heldout = {case["case_id"] for case in cases if case["split"] == "heldout"}
    assert len(development) == 12
    assert len(heldout) == 20
    assert development.isdisjoint(heldout)
    known = {ticket.ticket_id for ticket in load_history(ROOT / "starter_kit" / "data" / "tickets.json")}
    known.update(ticket.ticket_id for ticket in load_history(ROOT / "data" / "production_tickets.json"))
    assert all(set(case["relevant_ticket_ids"]).issubset(known) for case in cases)


def test_report_counts_false_suggestions_not_just_hits():
    history = load_history(ROOT / "data" / "production_tickets.json")
    index = SimilarityIndex(history, max_age_days=180, today=date(2026, 10, 31))
    report = evaluate(index, load_cases()[:1])
    assert report["case_count"] == 1
    assert report["retrieval_mode"] in {"hybrid", "lexical_fallback"}
    assert report["semantic_search_enabled"] is (index.semantic_matrix is not None)
    assert report["mean_recall_at_3"] == 1
    assert report["matching_p95_ms"] >= 0
    assert report["outcomes"][0]["returned_ids"][0] == "T-002003"

@pytest.mark.semantic
@pytest.mark.skipif(semantic_model_source() is None, reason="Download the model with: python -m support_assistant.semantic")
@pytest.mark.parametrize("split", ["development", "heldout"])
def test_hybrid_retrieval_meets_quality_bar(split):
    history = load_history(ROOT / "starter_kit" / "data" / "tickets.json") + load_history(ROOT / "data" / "production_tickets.json")
    index = SimilarityIndex(history, max_age_days=180, today=CORPUS_AS_OF)
    report = evaluate(index, [case for case in load_cases() if case["split"] == split])
    assert report["retrieval_mode"] == "hybrid"
    assert report["mean_recall_at_3"] == 1
    assert report["correct_abstention_rate"] == 1
    assert report["irrelevant_returned_count"] == 0
