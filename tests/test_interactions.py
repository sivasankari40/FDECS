import pytest

from support_assistant.interactions import InteractionHistory, category_codes
from support_assistant.models import TicketQuery


def record(history):
    return history.record(TicketQuery("Billing payment", "A duplicate card charge"), ["Billing"], ("T-001303",), "test-fingerprint")


def test_category_groups_include_dashboard_rendering_and_performance():
    assert category_codes(["Dashboard"]) == frozenset({"render", "perf"})
    assert category_codes(["Auth", "Billing", "Sync"]) == frozenset({"auth", "billing", "sync"})
    assert category_codes([]) is None


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError):
        category_codes(["Unknown"])


def test_history_is_bounded_and_newest_first():
    history = InteractionHistory(limit=2)
    oldest = record(history)
    middle = record(history)
    newest = record(history)
    assert [entry.search_id for entry in history.entries] == [newest.search_id, middle.search_id]
    with pytest.raises(ValueError):
        history.find(oldest.search_id)


def test_feedback_can_be_changed_and_cleared():
    history = InteractionHistory()
    entry = record(history)
    history.vote(entry.search_id, "T-001303", 1)
    assert entry.votes == {"T-001303": 1}
    history.vote(entry.search_id, "T-001303", 0)
    assert entry.votes == {"T-001303": 0}
    history.vote(entry.search_id, "T-001303", None)
    assert not entry.votes


def test_feedback_is_scoped_to_a_search_result():
    history = InteractionHistory()
    first = record(history)
    second = record(history)
    history.vote(first.search_id, "T-001303", 1)
    assert not second.votes
    with pytest.raises(ValueError):
        history.vote(first.search_id, "T-009999", 1)


@pytest.mark.parametrize("value", [True, "helpful", 2, -1])
def test_invalid_feedback_rejected(value):
    history = InteractionHistory()
    entry = record(history)
    with pytest.raises(ValueError):
        history.vote(entry.search_id, "T-001303", value)


def test_histories_do_not_share_state():
    first = InteractionHistory()
    second = InteractionHistory()
    entry = record(first)
    first.vote(entry.search_id, "T-001303", 1)
    assert not second.entries
    first.clear()
    assert not first.entries


def test_history_query_uses_existing_redaction():
    history = InteractionHistory()
    query = TicketQuery("Password reset", "demo@example.invalid api_key=DEMO_ONLY_NOT_A_KEY", ("http 403",))
    entry = history.record(query, [], (), "test-fingerprint")
    assert "demo@example.invalid" not in entry.query.description
    assert "DEMO_ONLY_NOT_A_KEY" not in entry.query.description
    assert entry.query.error_codes == ("HTTP 403",)