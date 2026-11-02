import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from support_assistant.config import Settings
from support_assistant.models import Ticket
from support_assistant.ticket_ids import canonical_ticket_id
from support_assistant.observability import event


class RepositoryError(RuntimeError):
    pass


READ_ONLY_VALIDATOR = """function(newDoc, oldDoc, userCtx, secObj) {
    if (userCtx.roles.indexOf('support_reader') !== -1) {
        throw({forbidden: 'Support reader identities cannot write'});
    }
    if (!newDoc._deleted && newDoc.type === 'ticket' && secObj.ticket_store === 'resolved' && newDoc.status !== 'resolved') {
        throw({forbidden: 'Historical database accepts resolved tickets only'});
    }
}"""


@dataclass(frozen=True)
class ImportReport:
    created: int
    updated: int
    unchanged: int
    skipped_deleted: int = 0


def load_history(path: Path) -> list[Ticket]:
    try:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Historical input must be a readable UTF-8 JSON file.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("tickets"), list):
        raise ValueError("Historical input must contain a tickets array.")
    tickets = []
    seen = set()
    for document in payload["tickets"]:
        if not isinstance(document, dict):
            raise ValueError("Every historical ticket must be a JSON object.")
        ticket = Ticket.from_dict(document)
        if ticket.ticket_id in seen:
            raise ValueError(f"Duplicate ticket ID: {ticket.ticket_id}.")
        seen.add(ticket.ticket_id)
        tickets.append(ticket)
    return tickets


class CouchRepository:
    # Service identities already verified in this process; avoids re-checking on every Streamlit rerun.
    verified_identities: set[tuple[str, str, str]] = set()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base = f"{settings.url.rstrip('/')}/{quote(settings.database, safe='')}"
        self.session = requests.Session()
        retries = Retry(total=2, backoff_factor=0.2, allowed_methods=frozenset({"GET", "HEAD"}),
                status_forcelist=(429, 502, 503, 504), raise_on_status=False,
                respect_retry_after_header=False)
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        if settings.username:
            self.session.auth = (settings.username, settings.password)

    def __enter__(self) -> "CouchRepository":
        return self

    def __exit__(self, *args: Any) -> None:
        self.session.close()

    def _request(self, method: str, suffix: str = "", allowed: tuple[int, ...] = (200,), server_path: str | None = None, **kwargs: Any) -> Any:
        start = perf_counter()
        if server_path is not None and server_path not in {"/_session", "/_users"} and not server_path.startswith("/_users/org.couchdb.user%3A"):
            raise ValueError("Unsupported administrative endpoint.")
        url = self.settings.url.rstrip("/") + server_path if server_path is not None else self.base + suffix
        try:
            response = self.session.request(method, url, timeout=(3, 10), allow_redirects=False, **kwargs)
        except requests.RequestException as error:
            event("database_unreachable", elapsed_ms=round((perf_counter() - start) * 1000, 2))
            raise RepositoryError("CouchDB is unreachable or timed out. Check the server and COUCHDB_URL.") from error
        event("database_request", status=response.status_code, elapsed_ms=round((perf_counter() - start) * 1000, 2))
        if response.status_code in {401, 403}:
            raise RepositoryError("CouchDB denied access. Configure COUCHDB_USER and COUCHDB_PASSWORD in .env, then restart the app.")
        if response.status_code not in allowed:
            if response.status_code == 404:
                raise RepositoryError("The ticket database does not exist. Run the documented ingestion command first.")
            raise RepositoryError(f"CouchDB request failed (HTTP {response.status_code}).")
        try:
            payload = response.json()
        except ValueError as error:
            raise RepositoryError("CouchDB returned an invalid JSON response.") from error
        expected = list if suffix == "/_bulk_docs" else dict
        if not isinstance(payload, expected):
            raise RepositoryError("CouchDB returned an unexpected response shape.")
        return payload

    def initialize(self) -> None:
        result = self._request("PUT", allowed=(201, 202, 412))
        if result.get("ok") and self.settings.username:
            self._request("PUT", "/_security", json={
                "admins": {"names": [self.settings.username], "roles": []},
                "members": {"names": [self.settings.username], "roles": []},
            })

    def info(self) -> dict[str, Any]:
        identity = (self.settings.url, self.settings.database, self.settings.username)
        if self.settings.environment == "production" and self.settings.role == "reader" and identity not in self.verified_identities:
            context = self._request("GET", server_path="/_session").get("userCtx", {})
            roles = context.get("roles", [])
            security = self._request("GET", "/_security")
            guard = self._request("GET", "/_design/support_access")
            if context.get("name") != self.settings.username or "support_reader" not in roles or "_admin" in roles:
                raise RepositoryError("Production service identity has missing or elevated roles.")
            admins = security.get("admins", {})
            if self.settings.username in admins.get("names", []) or set(roles).intersection(admins.get("roles", [])) or guard.get("validate_doc_update") != READ_ONLY_VALIDATOR:
                raise RepositoryError("Production requires a non-admin reader identity and the managed validation guard.")
            self.verified_identities.add(identity)
        result = self._request("GET")
        if "update_seq" not in result:
            raise RepositoryError("CouchDB database information is missing update_seq.")
        return result

    def provision_reader(self) -> None:
        self._provision_service(self.settings.reader_username, self.settings.reader_password, "support_reader")

    def enforce_store_kind(self, kind: str) -> None:
        if kind not in {"resolved", "incoming"}:
            raise ValueError("Unknown ticket database kind.")
        self.initialize()
        guard = self._request("GET", "/_design/support_access", allowed=(200, 404))
        if guard.get("_rev") and guard.get("managed_by") != "solari_support_assistant":
            raise RepositoryError("Existing access design is unmanaged; store policy was not changed.")
        document = {"_id": "_design/support_access", "managed_by": "solari_support_assistant", "validate_doc_update": READ_ONLY_VALIDATOR}
        if guard.get("_rev"):
            document["_rev"] = guard["_rev"]
        if guard.get("validate_doc_update") != READ_ONLY_VALIDATOR:
            self._request("PUT", "/_design/support_access", allowed=(201, 202), json=document)
        security = self._request("GET", "/_security")
        security["ticket_store"] = kind
        self._request("PUT", "/_security", json=security)

    def _provision_service(self, username: str, password: str, role: str) -> None:
        if not username or not password or username == self.settings.username:
            raise ValueError("Service provisioning requires distinct credentials from the administrator.")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", username):
            raise ValueError("Read service usernames must use 2-64 lowercase letters, digits, hyphens, or underscores.")
        self.initialize()
        users_database = self._request("GET", allowed=(200, 404), server_path="/_users")
        if users_database.get("error") == "not_found":
            self._request("PUT", allowed=(201, 202, 412), server_path="/_users")
        user_path = "/_users/" + quote("org.couchdb.user:" + username, safe="")
        user = self._request("GET", allowed=(200, 404), server_path=user_path)
        if user.get("error") == "not_found":
            self._request("PUT", allowed=(201, 202), server_path=user_path, json={
                "_id": "org.couchdb.user:" + username, "name": username, "type": "user",
                "roles": [role], "password": password,
            })
        elif user.get("roles") != [role]:
            raise RepositoryError("Existing reader account has different roles; provisioning refuses to overwrite it.")
        guard = self._request("GET", "/_design/support_access", allowed=(200, 404))
        if "_rev" in guard and guard.get("managed_by") != "solari_support_assistant":
            raise RepositoryError("Existing access design document is not managed by this application.")
        if guard.get("validate_doc_update") != READ_ONLY_VALIDATOR:
            document = {"_id": "_design/support_access", "managed_by": "solari_support_assistant", "validate_doc_update": READ_ONLY_VALIDATOR}
            if "_rev" in guard:
                document["_rev"] = guard["_rev"]
            self._request("PUT", "/_design/support_access", allowed=(201, 202), json=document)
        security = self._request("GET", "/_security")
        if username in security.get("admins", {}).get("names", []) or role in security.get("admins", {}).get("roles", []):
            raise RepositoryError("Read identity is already a database admin; remove that privilege before provisioning.")
        members = security.setdefault("members", {"names": [], "roles": []})
        members["names"] = sorted(set(members.get("names", [])) | {username})
        self._request("PUT", "/_security", json=security)
        event("reader_provisioned")

    @staticmethod
    def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
        rows = result.get("rows")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise RepositoryError("CouchDB returned invalid document rows.")
        return rows

    def _documents(self) -> list[dict[str, Any]]:
        documents = []
        start_key = "ticket:"
        while True:
            params: dict[str, Any] = {"include_docs": "true", "limit": 501,
                                      "startkey": json.dumps(start_key), "endkey": json.dumps("ticket:\uffff")}
            result = self._request("GET", "/_all_docs", params=params)
            rows = self._rows(result)
            for row in rows[:500]:
                document = row.get("doc")
                if document is not None and not isinstance(document, dict):
                    raise RepositoryError("CouchDB returned an invalid document.")
                if document and not row.get("value", {}).get("deleted"):
                    documents.append(document)
            if len(documents) > self.settings.max_tickets:
                raise RepositoryError("History exceeds MAX_HISTORY_TICKETS. Use a dedicated search tier or raise the tested capacity limit.")
            if len(rows) <= 500:
                break
            next_key = rows[500].get("key")
            if not isinstance(next_key, str) or next_key <= start_key:
                raise RepositoryError("CouchDB pagination did not advance.")
            start_key = next_key
        return documents

    def list_tickets(self) -> list[Ticket]:
        documents = [document for document in self._documents() if document.get("type") == "ticket"]
        try:
            tickets = [Ticket.from_dict(document) for document in documents]
            if any(document.get("_id") != "ticket:" + str(document.get("ticket_id")) for document in documents):
                raise ValueError("Stored document IDs do not match ticket IDs.")
            if len({ticket.ticket_id for ticket in tickets}) != len(tickets):
                raise ValueError("Stored ticket IDs are duplicated.")
            return tickets
        except ValueError as error:
            raise RepositoryError("Stored ticket data is invalid. Validate the import before refreshing the index.") from error

    def get_ticket(self, ticket_id: str) -> Ticket | None:
        if self.settings.environment == "production":
            self.info()  # Verifies the service identity before its first read.
        document = self._request("GET", "/" + quote("ticket:" + canonical_ticket_id(ticket_id), safe=""), allowed=(200, 404))
        if document.get("error") == "not_found" or document.get("type") != "ticket":
            return None
        try:
            return Ticket.from_dict(document)
        except ValueError as error:
            raise RepositoryError("Stored ticket data is invalid.") from error

    def import_tickets(self, tickets: list[Ticket]) -> ImportReport:
        ticket_ids = [ticket.ticket_id for ticket in tickets]
        if len(ticket_ids) != len(set(ticket_ids)):
            raise ValueError("Import contains duplicate ticket IDs.")
        if len(tickets) > self.settings.max_tickets:
            raise ValueError("Import exceeds MAX_HISTORY_TICKETS.")
        self.initialize()
        if tickets:
            policy = self._request("GET", "/_security").get("ticket_store")
            if policy == "resolved" and any(ticket.status != "resolved" for ticket in tickets):
                raise ValueError("Historical database accepts resolved tickets only; route incoming tickets to its separate database.")
        pending = []
        created = updated = unchanged = skipped_deleted = 0
        for offset in range(0, len(tickets), 500):
            batch = tickets[offset:offset + 500]
            keys = ["ticket:" + ticket.ticket_id for ticket in batch]
            rows = self._rows(self._request("POST", "/_all_docs", params={"include_docs": "true"}, json={"keys": keys}))
            existing = {row["key"]: row for row in rows}
            for ticket, document_id in zip(batch, keys):
                document = {"_id": document_id, "type": "ticket", **ticket.to_dict()}
                row = existing.get(document_id, {})
                previous = row.get("doc")
                if previous and not previous.get("_deleted"):
                    if previous.get("origin") == "incoming" and ticket.origin == "historical":
                        raise RepositoryError("Historical import would overwrite an incoming ticket. Resolve the ID collision explicitly.")
                    if {key: value for key, value in previous.items() if key != "_rev"} == document:
                        unchanged += 1
                        continue
                    document["_rev"] = previous["_rev"]
                    updated += 1
                elif row.get("value", {}).get("deleted"):
                    skipped_deleted += 1
                    continue
                else:
                    created += 1
                pending.append(document)
        for offset in range(0, len(pending), 500):
            batch = pending[offset:offset + 500]
            results = self._request("POST", "/_bulk_docs", allowed=(201, 202), json={"docs": batch})
            expected_ids = {document["_id"] for document in batch}
            failures = [result for result in results if not isinstance(result, dict) or "error" in result
                        or not isinstance(result.get("rev"), str) or result.get("id") not in expected_ids]
            returned_ids = {result.get("id") for result in results if isinstance(result, dict)}
            if failures or len(results) != len(batch) or returned_ids != expected_ids:
                conflicts = sum(isinstance(result, dict) and result.get("error") == "conflict" for result in failures)
                raise RepositoryError(f"Import partially failed: {len(failures)} document errors ({conflicts} revision conflicts). Successful writes remain stored; rerun ingestion to reconcile.")
        event("history_import", created=created, updated=updated, unchanged=unchanged, skipped_deleted=skipped_deleted)
        return ImportReport(created, updated, unchanged, skipped_deleted)

    def changes(self, since: Any) -> dict[str, Any]:
        result = self._request("GET", "/_changes", params={"since": since, "include_docs": "true", "limit": 500, "feed": "normal"})
        rows = result.get("results")
        pending = result.get("pending")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows) or "last_seq" not in result:
            raise RepositoryError("CouchDB returned an invalid changes feed.")
        if isinstance(pending, bool) or not isinstance(pending, int) or pending < 0:
            raise RepositoryError("CouchDB changes feed has invalid pending count.")
        return result
