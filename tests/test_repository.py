import json
import os
import secrets
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
import requests

from support_assistant.config import Settings
from support_assistant.repository import READ_ONLY_VALIDATOR, CouchRepository, RepositoryError, load_history
from support_assistant.service import IndexStore


DATA = Path(__file__).resolve().parents[1] / "starter_kit" / "data" / "tickets.json"


@pytest.fixture
def ticket():
    return load_history(DATA)[0]


def test_fixture_integrity():
    history = load_history(DATA)
    assert len(history) == 40
    assert len({ticket.ticket_id for ticket in history}) == 40


@pytest.mark.parametrize("change", [
    {"tags": "login"}, {"created_at": "2025-02-30"}, {"resolution_notes": ""},
    {"subject": None}, {"description": ""}, {"ticket_id": 1},
])
def test_invalid_history(tmp_path, ticket, change):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"tickets": [{**ticket.to_dict(), **change}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_history(path)


def test_duplicate_input(tmp_path, ticket):
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps({"tickets": [ticket.to_dict(), ticket.to_dict()]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_history(path)


def test_import_is_idempotent(ticket):
    with CouchRepository(Settings()) as repository:
        repository._request = Mock(side_effect=[{"error": "file_exists"}, {"ticket_store": "resolved"}, {"rows": [
            {"key": "ticket:" + ticket.ticket_id, "doc": {"_id": "ticket:" + ticket.ticket_id, "_rev": "1-a", "type": "ticket", **ticket.to_dict()}}
        ]}])
        report = repository.import_tickets([ticket])
        assert report.unchanged == 1
        assert report.created == report.updated == 0
        assert repository._request.call_count == 3


def test_updates_use_revision(ticket):
    with CouchRepository(Settings()) as repository:
        previous = {"_id": "ticket:" + ticket.ticket_id, "_rev": "1-a", "type": "ticket", **ticket.to_dict(), "subject": "Old subject"}
        repository._request = Mock(side_effect=[{"error": "file_exists"}, {"ticket_store": "resolved"}, {"rows": [{"key": previous["_id"], "doc": previous}]}, [{"ok": True, "id": previous["_id"], "rev": "2-b"}]])
        report = repository.import_tickets([ticket])
        assert report.updated == 1
        assert repository._request.call_args.kwargs["json"]["docs"][0]["_rev"] == "1-a"


def test_bulk_conflicts_report_partial_failure(ticket):
    with CouchRepository(Settings()) as repository:
        repository._request = Mock(side_effect=[{"error": "file_exists"}, {"ticket_store": "resolved"}, {"rows": []}, [{"error": "conflict"}]])
        with pytest.raises(RepositoryError, match="partially failed"):
            repository.import_tickets([ticket])


def test_deleted_ticket_is_not_resurrected_by_routine_import(ticket):
    with CouchRepository(Settings()) as repository:
        repository._request = Mock(side_effect=[{"error": "file_exists"}, {"ticket_store": "resolved"}, {"rows": [
            {"key": "ticket:" + ticket.ticket_id, "doc": None, "value": {"deleted": True, "rev": "2-deleted"}}
        ]}, [{"ok": True, "id": "ticket:" + ticket.ticket_id, "rev": "3-restored"}]])
        report = repository.import_tickets([ticket])
        assert report.updated == 0
        assert report.skipped_deleted == 1
        assert repository._request.call_count == 3


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_http_errors_are_actionable_and_do_not_expose_secrets(status):
    with CouchRepository(Settings(username="test-user", password="test-secret")) as repository:
        repository.session.request = Mock(return_value=Mock(status_code=status))
        with pytest.raises(RepositoryError) as caught:
            repository.info()
        assert "test-secret" not in str(caught.value)


def test_connection_failure():
    with CouchRepository(Settings()) as repository:
        repository.session.request = Mock(side_effect=requests.ConnectionError())
        with pytest.raises(RepositoryError, match="unreachable"):
            repository.info()


def test_pagination(ticket):
    rows = [{"key": f"ticket:T-{900000 + index:06d}", "doc": {"_id": f"ticket:T-{900000 + index:06d}", "type": "ticket", **replace(ticket, ticket_id=f"T-{900000 + index:06d}", legacy_ids=()).to_dict()}} for index in range(501)]
    with CouchRepository(Settings()) as repository:
        repository._request = Mock(side_effect=[{"rows": rows}, {"rows": rows[500:]}])
        assert len(repository.list_tickets()) == 501
        assert "skip" not in repository._request.call_args.kwargs["params"]
        assert repository._request.call_args.kwargs["params"]["startkey"] == '"ticket:T-900500"'


def test_history_capacity_is_bounded(ticket):
    document = {"_id": "ticket:" + ticket.ticket_id, "type": "ticket", **ticket.to_dict()}
    with CouchRepository(Settings(max_tickets=1)) as repository:
        repository._request = Mock(return_value={"rows": [{"doc": document}, {"doc": document}]})
        with pytest.raises(RepositoryError, match="MAX_HISTORY_TICKETS"):
            repository.list_tickets()


def test_nonadvancing_pagination_fails(ticket):
    with CouchRepository(Settings()) as repository:
        repository._request = Mock(return_value={"rows": [{"key": "ticket:", "doc": None}] * 501})
        with pytest.raises(RepositoryError, match="did not advance"):
            repository.list_tickets()


def test_reads_retry_but_writes_do_not():
    with CouchRepository(Settings()) as repository:
        retry = repository.session.get_adapter("http://").max_retries
        assert "GET" in retry.allowed_methods
        assert "POST" not in retry.allowed_methods
        assert retry.total == 2


def test_redirects_are_not_followed():
    with CouchRepository(Settings()) as repository:
        repository.session.request = Mock(return_value=Mock(status_code=302))
        with pytest.raises(RepositoryError, match="HTTP 302"):
            repository.info()
        assert repository.session.request.call_args.kwargs["allow_redirects"] is False


def test_malformed_json_shape_is_actionable():
    with CouchRepository(Settings()) as repository:
        response = Mock(status_code=200)
        response.json.return_value = []
        repository.session.request = Mock(return_value=response)
        with pytest.raises(RepositoryError, match="response shape"):
            repository.info()


def test_reader_database_admin_role_is_rejected():
    settings = Settings(environment="production", role="reader", username="synthetic-reader", password="test-only",
                        reader_username="synthetic-reader", reader_password="test-only", resolution_age_days=180,
                        allowed_emails=("agent@solari.example",), oidc_issuer="https://idp.example.invalid")
    with CouchRepository(settings) as repository:
        repository._request = Mock(side_effect=[{"userCtx": {"name": "synthetic-reader", "roles": ["support_reader", "db_admin"]}},
                                                {"admins": {"names": [], "roles": ["db_admin"]}},
                                                {"validate_doc_update": READ_ONLY_VALIDATOR}])
        with pytest.raises(RepositoryError, match="non-admin reader"):
            repository.info()


@pytest.mark.parametrize("kwargs", [
    {"url": "file:///tmp/database"}, {"url": "http://user:secret@localhost:5984"},
    {"database": "../other"}, {"username": "admin"},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_COUCHDB_TESTS") != "1", reason="Set RUN_COUCHDB_TESTS=1 to run against real CouchDB.")
def test_real_couchdb_roundtrip(ticket):
    settings = replace(Settings.from_env(), database="solari_test_" + uuid4().hex)
    with CouchRepository(settings) as repository:
        initialized = False
        try:
            first = repository.import_tickets([ticket])
            initialized = True
            second = repository.import_tickets([ticket])
            assert first.created == 1
            assert second.unchanged == 1
            assert repository.list_tickets() == [ticket]
            store = IndexStore()
            original_snapshot = store.refresh(repository)
            changed = replace(ticket, subject="Updated password reset incident")
            assert repository.import_tickets([changed]).updated == 1
            assert repository.list_tickets() == [changed]
            updated_snapshot = store.refresh(repository)
            assert updated_snapshot.tickets == (changed,)
            assert updated_snapshot.fingerprint != original_snapshot.fingerprint
            document = repository._request("GET", "/ticket:" + ticket.ticket_id)
            repository._request("DELETE", "/ticket:" + ticket.ticket_id, params={"rev": document["_rev"]})
            assert store.refresh(repository).tickets == ()
            assert repository.import_tickets([changed]).skipped_deleted == 1
            assert repository.list_tickets() == []
        finally:
            if initialized:
                repository._request("DELETE", allowed=(200, 202))


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("RUN_COUCHDB_TESTS") != "1", reason="Set RUN_COUCHDB_TESTS=1 to test real read-only enforcement.")
def test_real_read_only_account_can_read_but_cannot_write(ticket):
    identifier = uuid4().hex
    settings = replace(Settings.from_env(), environment="development", database="solari_test_" + identifier,
                       reader_username="solari_reader_" + identifier, reader_password=secrets.token_urlsafe(32))
    user_path = "/_users/org.couchdb.user%3A" + settings.reader_username
    with CouchRepository(settings) as repository:
        initialized = False
        try:
            repository.initialize()
            initialized = True
            repository.provision_reader()
            repository.import_tickets([ticket])
            with CouchRepository(settings.for_reader()) as reader:
                assert reader.list_tickets() == [ticket]
                response = reader.session.put(reader.base + "/ticket:UNAUTHORIZED", json={"type": "ticket"}, timeout=(3, 10), allow_redirects=False)
                assert response.status_code == 403
                response = reader.session.put(reader.base + "/_design/UNAUTHORIZED", json={"language": "javascript"}, timeout=(3, 10), allow_redirects=False)
                assert response.status_code == 403
            production = replace(settings, environment="production", resolution_age_days=180,
                                 allowed_emails=("agent@solari.example",), oidc_issuer="https://idp.example.invalid")
            with CouchRepository(production.for_reader()) as reader:
                assert "update_seq" in reader.info()
                assert (reader.settings.url, reader.settings.database, reader.settings.username) in CouchRepository.verified_identities
        finally:
            if initialized:
                user = repository._request("GET", allowed=(200, 404), server_path=user_path)
                if "_rev" in user:
                    repository._request("DELETE", allowed=(200, 202), server_path=user_path, params={"rev": user["_rev"]})
                repository._request("DELETE", allowed=(200, 202))

def test_incoming_lookup_reads_one_document(ticket):
    with CouchRepository(Settings()) as repository:
        repository._request = Mock(side_effect=[{"_id": "ticket:" + ticket.ticket_id, "type": "ticket", **ticket.to_dict()}, {"error": "not_found"}])
        assert repository.get_ticket(ticket.ticket_id.lower()) == ticket
        assert repository.get_ticket("T-999999") is None
        assert repository._request.call_args_list[0].args[:2] == ("GET", "/ticket%3A" + ticket.ticket_id)
