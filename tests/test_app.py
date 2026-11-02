from pathlib import Path
import json
from dataclasses import replace
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from support_assistant.repository import RepositoryError, load_history
from support_assistant.config import Settings
from support_assistant.models import Ticket


ROOT = Path(__file__).resolve().parents[1]


class FakeRepository:
    documents = load_history(ROOT / "starter_kit" / "data" / "tickets.json")
    unavailable = False
    created = []
    revision = 0
    incoming_unavailable = False
    incoming_documents = [Ticket.incoming("T-005003", "Getting charged twice this month", "Customer's card statement shows two charges for a single monthly payment.")]

    def __init__(self, settings):
        self.settings = settings

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def info(self):
        if self.unavailable or (self.settings.database == "solari_incoming_tickets" and self.incoming_unavailable):
            raise RepositoryError("CouchDB denied access. Configure credentials locally.")
        return {"update_seq": len(self.documents) + len(self.created) + self.revision}

    def list_tickets(self):
        if self.settings.database == "solari_incoming_tickets":
            return self.incoming_documents
        return self.documents + self.created

    def get_ticket(self, ticket_id):
        self.info()
        return next((ticket for ticket in self.list_tickets() if ticket.ticket_id == ticket_id), None)

    def changes(self, since):
        return {"results": [{"id": "ticket:" + ticket.ticket_id, "doc": {"_id": "ticket:" + ticket.ticket_id, "type": "ticket", **ticket.to_dict()}}
                            for ticket in self.list_tickets()], "last_seq": self.info()["update_seq"], "pending": 0}


class FakeFeedbackStore:
    votes = {}
    missed = {}
    unavailable = False

    def __init__(self, settings):
        if self.unavailable:
            raise RepositoryError("CouchDB is unreachable or timed out.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def record(self, search_id, ticket_id, incoming_ticket_id, helpful, rank, score):
        FakeFeedbackStore.votes[(search_id, ticket_id)] = {"incoming_ticket_id": incoming_ticket_id, "helpful": helpful, "rank": rank}

    def record_missed(self, search_id, ticket_id, incoming_ticket_id, shown_ticket_ids, reason, score):
        FakeFeedbackStore.missed[(search_id, ticket_id)] = {"incoming_ticket_id": incoming_ticket_id, "shown": tuple(shown_ticket_ids),
                                                            "reason": reason, "score": score}

    def counts(self, ticket_ids):
        totals = {}
        for (_, ticket_id), vote in self.votes.items():
            if ticket_id in ticket_ids and vote["helpful"] is not None:
                helpful, unhelpful = totals.get(ticket_id, (0, 0))
                totals[ticket_id] = (helpful + vote["helpful"], unhelpful + (not vote["helpful"]))
        return totals


@pytest.fixture(autouse=True)
def in_memory_feedback():
    # App tests must never write votes to a real CouchDB.
    FakeFeedbackStore.votes = {}
    FakeFeedbackStore.missed = {}
    with patch("support_assistant.feedback.FeedbackStore", FakeFeedbackStore):
        yield FakeFeedbackStore


def app():
    FakeRepository.created = []
    FakeRepository.revision = 0
    st.cache_resource.clear()
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()


def submit(test, example):
    examples = json.loads((ROOT / "starter_kit" / "data" / "new_tickets_example.json").read_text(encoding="utf-8"))["new_tickets"]
    ticket = next(ticket for ticket in examples if ticket["ticket_id"] == example)
    next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-005003").run()
    next(widget for widget in test.text_input if widget.label == "Subject").set_value(ticket["subject"])
    test.text_area[0].set_value(ticket["description"])
    next(button for button in test.button if button.label == "Find similar tickets").click().run()


def test_positive_search_renders_original_resolution():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-3")
        assert not test.exception
        assert any("T-001303" in caption.value for caption in test.caption)
        assert any("idempotency keys" in markdown.value for markdown in test.markdown)


def test_no_match_state():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-4")
        assert not test.exception
        assert test.info[0].value == "No strong historical match found."
        assert any("Closest match score:" in caption.value for caption in test.caption)
        assert any("Data Integration Team" in item.value for item in test.markdown)
        assert not any(button.label == "Use resolution" for button in test.button)


def test_invalid_input_state():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-005003").run()
        next(widget for widget in test.text_input if widget.label == "Subject").set_value("")
        next(button for button in test.button if button.label == "Find similar tickets").click().run()
        assert not test.exception
        assert any("subject must be a nonempty string" in error.value for error in test.error)


def test_ticket_id_is_required_for_search():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Subject").set_value("API request rejected")
        test.text_area[0].set_value("The API client receives HTTP 403 Forbidden.")
        next(button for button in test.button if button.label == "Find similar tickets").click().run()
        assert any("Ticket ID cannot be empty" in error.value for error in test.error)
        assert "results" not in test.session_state
        assert not test.exception


def test_unavailable_database_state():
    with patch("support_assistant.repository.CouchRepository", FakeRepository), patch.object(FakeRepository, "unavailable", True):
        test = app()
        assert not test.exception
        assert any("denied access" in error.value for error in test.error)


def test_empty_history_state():
    with patch("support_assistant.repository.CouchRepository", FakeRepository), patch.object(FakeRepository, "documents", []):
        test = app()
        assert not test.exception
        assert test.warning[0].value == "Ticket history is empty."


def test_refresh_clears_previous_results():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-2")
        next(button for button in test.button if button.label == "Refresh history").click().run()
        assert not test.exception
        assert not any("T-001201" in caption.value for caption in test.caption)


def test_search_rate_limit():
    settings = replace(Settings.from_env(), search_limit_per_minute=1)
    with patch("support_assistant.repository.CouchRepository", FakeRepository), patch("support_assistant.config.Settings.from_env", return_value=settings):
        test = app()
        submit(test, "NT-2")
        submit(test, "NT-3")
        assert not test.exception
        assert any("Search limit reached" in error.value for error in test.error)


def test_production_access_denied_before_database_read():
    settings = Settings(environment="production", username="synthetic-importer", password="test-only",
                        reader_username="synthetic-reader", reader_password="test-only",
                        allowed_emails=("agent@solari.example",), oidc_issuer="https://idp.example.invalid", resolution_age_days=180)
    auth = {"client_id": "synthetic", "client_secret": "synthetic", "cookie_secret": "x" * 32,
            "redirect_uri": "https://support.example.invalid/oauth2callback", "server_metadata_url": "https://idp.example.invalid"}
    with patch("support_assistant.config.Settings.from_env", return_value=settings), patch("streamlit.secrets", {"auth": auth}), patch("streamlit.user", {"is_logged_in": True, "email": "outsider@solari.example"}), patch("support_assistant.repository.CouchRepository") as repository:
        test = app()
        assert not test.exception
        assert any("Access denied" in error.value for error in test.error)
        repository.assert_not_called()


def test_incoming_ticket_is_direct_entry_without_dropdown():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        assert not test.selectbox
        assert [widget.label for widget in test.text_input] == ["Ticket ID", "Subject", "Error code (optional)"]
        assert test.text_area[0].label == "Description"
        assert not test.exception


def test_search_history_reopens_query_and_clears():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-3")
        assert any(button.label == "Getting charged twice this month" for button in test.button)
        submit(test, "NT-2")
        next(button for button in test.button if button.label == "Getting charged twice this month").click().run()
        assert next(widget for widget in test.text_input if widget.label == "Subject").value == "Getting charged twice this month"
        assert any("T-001303" in caption.value for caption in test.caption)
        next(button for button in test.button if button.label == "Clear search history").click().run()
        assert not test.session_state.interactions.entries
        assert not any("T-001303" in caption.value for caption in test.caption)
        assert not test.exception


def test_category_change_reruns_current_query():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-3")
        test.session_state.categories = ["Auth"]
        test.run()
        assert test.info[0].value == "No strong historical match found."
        test.session_state.categories = ["Billing"]
        test.run()
        assert any("T-001303" in caption.value for caption in test.caption)
        assert not test.exception


def test_feedback_is_saved_for_current_result():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-3")
        test.feedback[0].set_value(1).run()
        entry = test.session_state.interactions.find(test.session_state.current_search_id)
        assert entry.votes["T-001303"] == 1
        assert any("Marked helpful." in caption.value for caption in test.caption)
        test.feedback[0].set_value(0).run()
        assert entry.votes["T-001303"] == 0
        test.feedback[0].set_value(None).run()
        assert "T-001303" not in entry.votes
        assert not test.exception


def test_result_explanation_badges_and_percentage_similarity():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-3")
        assert any(heading.value == "Top 3 similar tickets" for heading in test.subheader)
        assert any('class="term-chip"' in item.value for item in test.markdown)
        assert any("Historical category: Billing" in item.value for item in test.markdown)
        assert any(metric.label == "Evidence score" and "%" in metric.value for metric in test.metric)
        assert not test.exception


def test_error_code_input_boosts_exact_historical_match():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-005003").run()
        next(widget for widget in test.text_input if widget.label == "Subject").set_value("API request fails")
        test.text_area[0].set_value("The API endpoint returns an error for a valid request.")
        next(widget for widget in test.text_input if widget.label == "Error code (optional)").set_value("HTTP 500")
        next(button for button in test.button if button.label == "Find similar tickets").click().run()
        assert test.session_state.results[0].ticket.ticket_id == "T-001703"
        assert test.session_state.results[0].matched_error_codes == ("HTTP 500",)
        assert any("Exact error-code match: HTTP 500" in item.value for item in test.markdown)
        assert not test.exception


def test_reuse_creates_editable_draft_and_clear_removes_it():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-3")
        next(button for button in test.button if button.label == "Use resolution").click().run()
        assert test.session_state.draft_source == "T-001303"
        assert "idempotency keys" in test.session_state.resolution_draft
        draft_widget = next(widget for widget in test.text_area if widget.label == "Draft text")
        draft_widget.set_value("Agent-reviewed synthetic draft").run()
        assert test.code[0].value == "Agent-reviewed synthetic draft"
        next(button for button in test.button if button.label == "Clear search history").click().run()
        assert not any(widget.label == "Draft text" for widget in test.text_area)
        assert not test.exception


def test_ticket_id_automatically_populates_subject_and_description():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-005003").run()
        ticket = FakeRepository.incoming_documents[0]
        assert next(widget for widget in test.text_input if widget.label == "Subject").value == ticket.subject
        assert test.text_area[0].value == ticket.description
        next(button for button in test.button if button.label == "Find similar tickets").click().run()
        assert all(match.ticket.ticket_id != "T-005003" for match in test.session_state.results)
        assert any(match.ticket.ticket_id == "T-001303" for match in test.session_state.results)
        assert not test.exception


def test_unknown_ticket_id_does_not_leave_previous_ticket_text():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-005003").run()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-999999").run()
        assert next(widget for widget in test.text_input if widget.label == "Subject").value == ""
        assert test.text_area[0].value == ""
        assert any("was not found" in warning.value for warning in test.warning)
        assert not test.exception


def test_ticket_id_lookup_keeps_fields_editable_and_optional():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-005003").run()
        next(widget for widget in test.text_input if widget.label == "Subject").set_value("Edited billing issue").run()
        assert next(widget for widget in test.text_input if widget.label == "Subject").value == "Edited billing issue"
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("").run()
        assert next(widget for widget in test.text_input if widget.label == "Subject").value == "Edited billing issue"
        assert not test.exception


def test_invalid_ticket_id_is_rejected():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("../other").run()
        assert any("T-######" in warning.value for warning in test.warning)
        assert not test.exception


def test_history_replay_retains_loaded_ticket_id_and_self_exclusion():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-005003").run()
        next(button for button in test.button if button.label == "Find similar tickets").click().run()
        subject = next(widget for widget in test.text_input if widget.label == "Subject").value
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("").run()
        next(button for button in test.button if button.label == subject).click().run()
        assert next(widget for widget in test.text_input if widget.label == "Ticket ID").value == "T-005003"
        assert all(match.ticket.ticket_id != "T-005003" for match in test.session_state.results)
        assert not test.exception


def test_manual_search_never_persists_a_ticket():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-4")
        assert next(widget for widget in test.text_input if widget.label == "Ticket ID").value == "T-005003"
        assert not FakeRepository.created
        assert not test.exception


def test_no_save_or_incoming_category_controls():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        assert not any(button.label == "Save ticket" for button in test.button)
        assert not any(group.label == "Ticket category" for group in test.button_group)
        next(button for button in test.button if button.label == "Clear ticket").click().run()
        assert next(widget for widget in test.text_input if widget.label == "Ticket ID").value == ""
        assert next(widget for widget in test.text_input if widget.label == "Subject").value == ""
        assert not test.exception


def test_historical_id_is_not_loaded_as_an_incoming_ticket():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-001303").run()
        assert any("incoming-ticket database" in warning.value for warning in test.warning)
        assert next(widget for widget in test.text_input if widget.label == "Subject").value == ""
        assert not test.exception


def test_unregistered_legacy_alias_does_not_load_an_existing_new_ticket():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("NT-3").run()
        assert next(widget for widget in test.text_input if widget.label == "Subject").value == ""
        assert any("not registered" in warning.value for warning in test.warning)
        assert next(button for button in test.button if button.label == "Find similar tickets").disabled
        assert not test.exception


def test_incoming_outage_does_not_block_manual_historical_search():
    with patch("support_assistant.repository.CouchRepository", FakeRepository), patch.object(FakeRepository, "incoming_unavailable", True):
        test = app()
        submit(test, "NT-3")
        assert any(match.ticket.ticket_id == "T-001303" for match in test.session_state.results)
        assert any("Incoming-ticket lookup is unavailable" in warning.value for warning in test.warning)
        assert not test.exception


def test_mixed_historical_database_blocks_search():
    mixed = FakeRepository.documents + [FakeRepository.incoming_documents[0]]
    with patch("support_assistant.repository.CouchRepository", FakeRepository), patch.object(FakeRepository, "documents", mixed):
        test = app()
        assert any("contains nonresolved tickets" in error.value for error in test.error)
        assert next(button for button in test.button if button.label == "Find similar tickets").disabled
        assert not test.exception


def test_error_code_input_surfaces_exact_matching_code():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-005003").run()
        next(widget for widget in test.text_input if widget.label == "Subject").set_value("API request fails with server error")
        test.text_area[0].set_value("A valid request receives an unexpected server-side failure from an API endpoint.")
        next(widget for widget in test.text_input if widget.label == "Error code (optional)").set_value("http 500")
        next(button for button in test.button if button.label == "Find similar tickets").click().run()
        assert test.session_state.results[0].ticket.ticket_id == "T-001703"
        assert test.session_state.results[0].matched_error_codes == ("HTTP 500",)
        assert any("Exact error-code match: HTTP 500" in item.value for item in test.markdown)
        assert not test.exception


def test_invalid_error_code_shows_validation_error():
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        next(widget for widget in test.text_input if widget.label == "Ticket ID").set_value("T-005003").run()
        next(widget for widget in test.text_input if widget.label == "Subject").set_value("API request failure")
        test.text_area[0].set_value("The endpoint returns an error.")
        next(widget for widget in test.text_input if widget.label == "Error code (optional)").set_value("not a code")
        next(button for button in test.button if button.label == "Find similar tickets").click().run()
        assert any("Use HTTP 4xx/5xx" in error.value for error in test.error)
        assert not test.exception

def test_votes_are_persisted_and_shown_to_later_searches(in_memory_feedback):
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-3")
        test.feedback[0].set_value(1).run()
        first_search = test.session_state.current_search_id
        assert in_memory_feedback.votes[(first_search, "T-001303")] == {"incoming_ticket_id": "T-005003", "helpful": True, "rank": 1}
        test.feedback[0].set_value(None).run()
        assert in_memory_feedback.votes[(first_search, "T-001303")]["helpful"] is None
        test.feedback[0].set_value(1).run()
        next(button for button in test.button if button.label == "Find similar tickets").click().run()
        assert any("Earlier agent feedback: helpful 1 | not helpful 0" in caption.value for caption in test.caption)
        assert not test.exception


def test_feedback_outage_keeps_the_session_vote(in_memory_feedback):
    with patch("support_assistant.repository.CouchRepository", FakeRepository), patch.object(in_memory_feedback, "unavailable", True):
        test = app()
        submit(test, "NT-3")
        test.feedback[0].set_value(1).run()
        entry = test.session_state.interactions.find(test.session_state.current_search_id)
        assert entry.votes["T-001303"] == 1
        assert any("kept for this session only" in warning.value for warning in test.warning)
        assert not test.exception


def test_differently_resolved_precedents_show_a_consistency_warning():
    from support_assistant.consistency import ResolutionConflict

    def one_conflict(index, tickets):
        return (ResolutionConflict(tickets[0], tickets[0], 0.7, 0.05),) if tickets else ()

    with patch("support_assistant.repository.CouchRepository", FakeRepository), patch("support_assistant.consistency.conflicts_among", one_conflict):
        test = app()
        submit(test, "NT-3")
        assert any("were resolved differently" in warning.value for warning in test.warning)
        assert not test.exception


def test_invalid_triage_configuration_is_visible(tmp_path, monkeypatch):
    broken = tmp_path / "routing.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("TRIAGE_ROUTING_FILE", str(broken))
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        assert any("Triage routing file" in error.value for error in test.error)
        assert not test.exception


def report_missed(test, ticket_id):
    next(widget for widget in test.text_input if widget.label == "Ticket ID of the right precedent").set_value(ticket_id)
    next(button for button in test.button if button.label == "Report missed precedent").click().run()


def test_missed_precedent_is_recorded_with_the_reason_it_was_hidden(in_memory_feedback):
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-3")
        search_id = test.session_state.current_search_id
        report_missed(test, "t-1302")
        report = in_memory_feedback.missed[(search_id, "T-001302")]
        assert report["incoming_ticket_id"] == "T-005003" and report["reason"] == "below_threshold"
        assert "T-001303" in report["shown"]
        assert any("Recorded T-001302 as a missed precedent" in item.value and "below the 30% threshold" in item.value for item in test.success)
        assert not test.exception


def test_missed_precedent_validation(in_memory_feedback):
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-3")
        report_missed(test, "T-001303")
        assert any("already shown above" in error.value for error in test.error)
        report_missed(test, "T-999999")
        assert any("not in the resolved ticket history" in error.value for error in test.error)
        report_missed(test, "nonsense")
        assert any("for example T-002001" in error.value for error in test.error)
        assert in_memory_feedback.missed == {}
        assert not test.exception


def test_missed_precedent_can_be_reported_after_a_no_match(in_memory_feedback):
    with patch("support_assistant.repository.CouchRepository", FakeRepository):
        test = app()
        submit(test, "NT-4")
        report_missed(test, "T-001101")
        (report,) = in_memory_feedback.missed.values()
        assert report["shown"] == () and report["reason"] == "below_threshold"
        assert not test.exception
