import os
import secrets
from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import requests

from support_assistant.config import Settings
from support_assistant.feedback import FeedbackStore, prepare_feedback_database
from support_assistant.repository import CouchRepository, RepositoryError


SEARCH_ID = "a" * 32


def store_with(responses):
    store = FeedbackStore(Settings())
    store.repository._request = Mock(side_effect=responses)
    return store


def test_new_vote_stores_ids_and_rank_but_no_ticket_text():
    store = store_with([{"error": "not_found"}, {"ok": True}])
    store.record(SEARCH_ID, "t-1303", "NT-3", True, 1, 0.6612345)
    method, path = store.repository._request.call_args_list[1].args[:2]
    document = store.repository._request.call_args_list[1].kwargs["json"]
    assert (method, path) == ("PUT", "/feedback%3A" + SEARCH_ID + "%3AT-001303")
    assert document["incoming_ticket_id"] == "T-005003"
    assert (document["helpful"], document["rank"], document["score"]) == (True, 1, 0.6612)
    assert set(document) == {"_id", "type", "search_id", "ticket_id", "incoming_ticket_id", "helpful", "rank", "score", "voted_at"}


def test_changed_vote_updates_the_same_document_and_retries_conflicts():
    store = store_with([{"_rev": "1-a"}, {"error": "conflict"}, {"_rev": "2-b"}, {"ok": True}])
    store.record(SEARCH_ID, "T-001303", None, None, 2, 0.4)
    assert store.repository._request.call_args_list[3].kwargs["json"]["_rev"] == "2-b"
    assert store.repository._request.call_args_list[3].kwargs["json"]["helpful"] is None


def test_unknown_incoming_ids_are_stored_as_null():
    store = store_with([{"error": "not_found"}, {"ok": True}])
    store.record(SEARCH_ID, "T-001303", "not-a-ticket", False, 3, 0.3)
    assert store.repository._request.call_args_list[1].kwargs["json"]["incoming_ticket_id"] is None


def test_counts_read_the_grouped_vote_view():
    store = store_with([{"rows": [{"key": "T-001303", "value": [3, 1]}]}])
    assert store.counts(["T-001303", "T-002010"]) == {"T-001303": (3, 1)}
    assert store.repository._request.call_args.kwargs["json"] == {"keys": ["T-001303", "T-002010"]}


def test_invalid_search_id_is_rejected():
    with pytest.raises(ValueError):
        store_with([]).record("not-a-search-id", "T-001303", None, True, 1, 0.5)


def test_feedback_database_name_must_be_separate():
    with pytest.raises(ValueError):
        Settings(feedback_database="solari_support_tickets")


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_COUCHDB_TESTS") != "1", reason="Set RUN_COUCHDB_TESTS=1 for real feedback persistence checks.")
def test_real_feedback_roundtrip_and_reader_write_guard():
    identifier = uuid4().hex
    settings = replace(Settings.from_env(), feedback_database="solari_test_feedback_" + identifier,
                       reader_username="solari_reader_" + identifier, reader_password=secrets.token_urlsafe(32))
    user_path = "/_users/org.couchdb.user%3A" + settings.reader_username
    with CouchRepository(settings.for_feedback(admin=True)) as admin:
        try:
            admin._request("PUT", allowed=(201, 202, 412), server_path="/_users")
            admin._request("PUT", allowed=(201, 202), server_path=user_path, json={
                "_id": "org.couchdb.user:" + settings.reader_username, "name": settings.reader_username,
                "type": "user", "roles": ["support_reader"], "password": settings.reader_password})
            prepare_feedback_database(settings)
            prepare_feedback_database(settings)  # idempotent
            with FeedbackStore(settings) as store:  # the non-admin reader identity
                store.record(SEARCH_ID, "T-001303", "T-005003", True, 1, 0.66)
                store.record("b" * 32, "T-001303", "T-005003", True, 1, 0.61)
                store.record("c" * 32, "T-001303", None, False, 2, 0.4)
                store.record("c" * 32, "T-001303", None, None, 2, 0.4)  # cleared votes are not counted
                assert store.counts(["T-001303", "T-002010"]) == {"T-001303": (2, 0)}
                assert len(store.votes()) == 2
                store.record_missed(SEARCH_ID, "T-002010", "T-005003", ["T-001303"], "below_threshold", 0.24)
                assert [(report["ticket_id"], report["reason"]) for report in store.missed()] == [("T-002010", "below_threshold")]
                forged = {"_id": "missed:" + "d" * 32 + ":T-002010", "type": "missed_precedent", "search_id": "d" * 32,
                          "ticket_id": "T-002010", "incoming_ticket_id": None, "shown_ticket_ids": [], "reason": "agent opinion",
                          "score": None, "reported_at": "2026-11-02T10:00:00+00:00", "note": "free text"}
                rejected = store.repository.session.put(store.repository.base + "/missed%3A" + "d" * 32 + "%3AT-002010", json=forged, timeout=(3, 10))
                assert rejected.status_code == 403
                base = store.repository.base
                document = store.repository._request("GET", "/feedback%3A" + SEARCH_ID + "%3AT-001303")
                session = store.repository.session
                deleted = session.delete(base + "/feedback%3A" + SEARCH_ID + "%3AT-001303", params={"rev": document["_rev"]}, timeout=(3, 10))
                assert deleted.status_code == 403
                invalid = session.put(base + "/feedback%3Aforged", json={"type": "feedback", "note": "free text"}, timeout=(3, 10))
                assert invalid.status_code == 403
                current_design = admin._request("GET", "/_design/feedback")
                design = session.put(base + "/_design/feedback", json={"_rev": current_design["_rev"], "views": {}}, timeout=(3, 10))
                assert design.status_code in {401, 403}
        finally:
            user = admin._request("GET", allowed=(200, 404), server_path=user_path)
            if "_rev" in user:
                admin._request("DELETE", allowed=(200, 202), server_path=user_path, params={"rev": user["_rev"]})
            admin._request("DELETE", allowed=(200, 202, 404))


def test_unreachable_feedback_database_raises_repository_error():
    store = FeedbackStore(Settings(url="http://127.0.0.1:9"))
    store.repository.session.request = Mock(side_effect=requests.ConnectionError())
    with pytest.raises(RepositoryError):
        store.counts(["T-001303"])
