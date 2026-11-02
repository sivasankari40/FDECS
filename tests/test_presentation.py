import json
from pathlib import Path

import pytest

from support_assistant.matching import SimilarityIndex
from support_assistant.models import TicketQuery
from support_assistant.presentation import load_triage_routing, match_reasons, no_match_advice, resolution_draft, term_badges
from support_assistant.config import Settings
from support_assistant.repository import load_history


DATA = Path(__file__).resolve().parents[1] / "starter_kit" / "data" / "tickets.json"


def match():
    index = SimilarityIndex(load_history(DATA))
    return index.search(TicketQuery("Charged twice", "Two charges on my card for a single month"))[0]


def test_term_badges_escape_markup_and_attributes():
    html = term_badges(("revenue", '<img src=x onerror="unsafe">', '" onclick="unsafe'))
    assert '<img' not in html
    assert '&lt;img' in html
    assert '&quot;' in html
    assert html.count('class="term-chip"') == 3


def test_explanation_uses_only_observed_evidence():
    reasons = match_reasons(match(), ("Billing",), 0.30)
    assert any("Within your selected category filter: Billing" in reason for reason in reasons)
    assert any("Historical category: Billing" in reason for reason in reasons)
    assert any("same root cause" in reason for reason in reasons)
    assert not any("Same category" in reason for reason in match_reasons(match(), (), 0.30))


def test_explanation_separates_text_tags_and_exact_codes():
    from dataclasses import replace

    selected = replace(match(), matched_tags=("payment-method",), matched_error_codes=("HTTP 403",), semantic_score=0.83)
    reasons = match_reasons(selected, (), 0.30)
    assert any("Shared subject/description terms" in reason for reason in reasons)
    assert any("Matching historical tags: payment-method" in reason for reason in reasons)
    assert any("Exact error-code match: HTTP 403" in reason for reason in reasons)
    assert any("Sentence Transformer semantic similarity: 83.0%" in reason for reason in reasons)


def test_snowflake_triage_does_not_require_a_weak_resolution():
    advice = no_match_advice(TicketQuery("Snowflake integration failing", "The new warehouse connector has an unclear error"), (),
                             load_triage_routing(Settings().triage_routing_file))
    assert advice.team == "Data Integration Team"
    assert any("knowledge-base article" in action for action in advice.actions)


def test_filtered_no_match_suggests_reviewing_filters():
    advice = no_match_advice(TicketQuery("Billing", "Duplicate charges"), ("Auth",), load_triage_routing(Settings().triage_routing_file))
    assert advice.actions[0].startswith("Review the category filters")


def test_reuse_is_a_provenance_labeled_draft_not_a_sent_reply():
    selected = match()
    draft = resolution_draft(selected)
    assert selected.ticket.ticket_id in draft
    assert selected.ticket.resolution_notes in draft

def write_routing(tmp_path, payload):
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_triage_routing_comes_from_configuration(tmp_path):
    path = write_routing(tmp_path, {"default_team": "Tier 1", "routes": [
        {"team": "Warehouse Squad", "keywords": ["Snowflake"]}, {"team": "Data Integration Team", "keywords": ["snowflake", "sync"]}]})
    routing = load_triage_routing(path)
    query = TicketQuery("Snowflake integration failing", "The connector has an unclear error")
    assert no_match_advice(query, (), routing).team == "Warehouse Squad"  # first matching route wins; keywords are case-insensitive
    assert no_match_advice(TicketQuery("Odd request", "Something else entirely"), (), routing).team == "Tier 1"


@pytest.mark.parametrize("payload", [
    [], {"routes": []}, {"default_team": "Tier 1"}, {"default_team": "Tier 1", "routes": [{"team": "X", "keywords": []}]},
    {"default_team": "Tier 1", "routes": [{"team": "X", "keywords": ["two words"]}]},
])
def test_invalid_triage_routing_is_rejected(tmp_path, payload):
    with pytest.raises(ValueError):
        load_triage_routing(write_routing(tmp_path, payload))


def test_triage_routing_file_is_configurable(monkeypatch, tmp_path):
    path = write_routing(tmp_path, {"default_team": "Tier 1", "routes": []})
    monkeypatch.setenv("TRIAGE_ROUTING_FILE", str(path))
    assert Settings.from_env().triage_routing_file == path
