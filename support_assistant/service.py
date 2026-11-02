import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock, Thread
from time import perf_counter
from typing import Any

import numpy as np

from support_assistant.config import ROOT
from support_assistant.models import Ticket
from support_assistant.ticket_ids import canonical_ticket_id
from support_assistant.matching import SimilarityIndex, search_text, text_key
from support_assistant.semantic import semantic_model_source
from support_assistant.observability import event
from support_assistant.repository import CouchRepository, RepositoryError


@dataclass(frozen=True)
class Snapshot:
    tickets: tuple[Ticket, ...]
    sequence: Any
    fingerprint: str
    change_sequence: Any = None


def make_snapshot(tickets: tuple[Ticket, ...], sequence: Any, change_sequence: Any = None) -> Snapshot:
    ordered = tuple(sorted(tickets, key=lambda ticket: ticket.ticket_id))
    serialized = json.dumps([ticket.to_dict() for ticket in ordered], sort_keys=True, ensure_ascii=True)
    return Snapshot(ordered, sequence, hashlib.sha256(serialized.encode("utf-8")).hexdigest(), change_sequence)


def read_snapshot(repository: CouchRepository) -> Snapshot:
    for attempt in range(3):
        before = repository.info()["update_seq"]
        tickets = tuple(sorted(repository.list_tickets(), key=lambda ticket: ticket.ticket_id))
        after = repository.info()["update_seq"]
        if before == after:
            return make_snapshot(tickets, after)
    raise RepositoryError("Ticket history changed during refresh. Try again after ingestion finishes.")


def read_incremental_snapshot(repository: CouchRepository, previous: Snapshot) -> Snapshot:
    documents = {"ticket:" + ticket.ticket_id: ticket for ticket in previous.tickets}
    sequence = previous.change_sequence if previous.change_sequence is not None else previous.sequence
    processed = 0
    for attempt in range(3):
        before = repository.info()["update_seq"]
        while True:
            changes = repository.changes(sequence)
            processed += len(changes["results"])
            if processed > repository.settings.max_tickets * 2:
                raise RepositoryError("Incremental refresh exceeds the bounded change budget. Refresh history explicitly.")
            for change in changes["results"]:
                document_id = change.get("id")
                if not isinstance(document_id, str):
                    raise RepositoryError("CouchDB changes feed is missing a document ID.")
                if not document_id.startswith("ticket:"):
                    continue
                if change.get("deleted"):
                    documents.pop("ticket:" + canonical_ticket_id(document_id.removeprefix("ticket:")), None)
                    continue
                document = change.get("doc")
                if not isinstance(document, dict) or document.get("_id") != document_id:
                    raise RepositoryError("Changed ticket document is invalid.")
                if document.get("type") != "ticket":
                    documents.pop(document_id, None)
                    continue
                try:
                    ticket = Ticket.from_dict(document)
                except ValueError as error:
                    raise RepositoryError("Changed ticket data is invalid; the previous snapshot was not published.") from error
                if document_id != "ticket:" + str(document.get("ticket_id")):
                    raise RepositoryError("Changed ticket ID does not match its document ID.")
                documents["ticket:" + ticket.ticket_id] = ticket
                if len(documents) > repository.settings.max_tickets:
                    raise RepositoryError("History exceeds MAX_HISTORY_TICKETS.")
            next_sequence = changes["last_seq"]
            if changes["pending"] and next_sequence == sequence:
                raise RepositoryError("CouchDB changes feed did not advance.")
            sequence = next_sequence
            if not changes["pending"]:
                break
        after = repository.info()["update_seq"]
        if before == after:
            event("incremental_refresh", changed_documents=processed, ticket_count=len(documents))
            return make_snapshot(tuple(documents.values()), after, sequence)
    raise RepositoryError("Ticket history changed during refresh. Try again after ingestion finishes.")


EMBEDDING_CACHE_DIRECTORY = ROOT / ".cache" / "embeddings"


class IndexStore:
    def __init__(self, cache_directory: Path | None = None, background: bool = False) -> None:
        self.lock = RLock()
        self.build_lock = Lock()
        self.snapshot: Snapshot | None = None
        self.index: SimilarityIndex | None = None
        self.index_key: tuple | None = None
        self.embedding_model: str | None = None
        self.embeddings: dict[str, np.ndarray] = {}
        # Embeddings persisted here survive restarts, so a cold start only encodes tickets changed since the last run.
        self.cache_directory = cache_directory
        # With background rebuilds, the previous index keeps serving searches while a new one is built.
        self.background = background
        self.building_key: tuple | None = None
        self.builder: Thread | None = None

    @property
    def rebuilding(self) -> bool:
        return self.building_key is not None

    def refresh(self, repository: CouchRepository, force: bool = False) -> Snapshot:
        with self.lock:
            sequence = repository.info()["update_seq"]
            if force or self.snapshot is None:
                candidate = read_snapshot(repository)
            elif self.snapshot.sequence != sequence:
                candidate = read_incremental_snapshot(repository, self.snapshot)
            else:
                return self.snapshot
            self.snapshot = candidate
            return candidate

    def index_for(self, snapshot: Snapshot) -> SimilarityIndex:
        key = (snapshot.fingerprint, semantic_model_source())
        with self.lock:
            if self.index is not None and self.index_key == key:
                return self.index
            if self.index is not None and self.background:
                if self.building_key != key:
                    self.building_key = key
                    self.builder = Thread(target=self._build_in_background, args=(snapshot, key), daemon=True)
                    self.builder.start()
                return self.index
        self._build(snapshot, key)
        return self.index

    def _build_in_background(self, snapshot: Snapshot, key: tuple) -> None:
        try:
            self._build(snapshot, key)
        except Exception:
            # Keep serving the previous index; the next request schedules another attempt.
            event("index_rebuild_failed")
            with self.lock:
                if self.building_key == key:
                    self.building_key = None

    def _build(self, snapshot: Snapshot, key: tuple) -> None:
        model = key[1]
        with self.build_lock:
            start = perf_counter()
            embeddings = self._embeddings_for(model)
            loaded = set(embeddings)
            index = SimilarityIndex(snapshot.tickets, embedding_cache=embeddings)
            index.fingerprint = snapshot.fingerprint
            # Keep only embeddings for currently indexed text so deleted or edited tickets do not accumulate.
            live = {text_key(search_text(ticket.subject, ticket.description)) for ticket in index.indexed}
            embeddings = {digest: vector for digest, vector in embeddings.items() if digest in live}
            if index.semantic_matrix is not None and set(embeddings) != loaded:
                self._persist(model, embeddings)
            with self.lock:
                self.embeddings, self.embedding_model = embeddings, model
                self.index, self.index_key = index, key
                if self.building_key == key:
                    self.building_key = None
            event("index_rebuild", ticket_count=len(index.indexed), elapsed_ms=round((perf_counter() - start) * 1000, 2))

    def _cache_file(self, model: str) -> Path | None:
        if self.cache_directory is None or model is None:
            return None
        return self.cache_directory / (hashlib.sha256(model.encode("utf-8")).hexdigest()[:16] + ".npz")

    def _embeddings_for(self, model: str | None) -> dict[str, np.ndarray]:
        if model == self.embedding_model and self.embeddings:
            return dict(self.embeddings)
        path = self._cache_file(model)
        if path is None or not path.is_file():
            return {}
        try:
            with np.load(path, allow_pickle=False) as stored:
                return dict(zip((str(key) for key in stored["keys"]), stored["vectors"]))
        except (OSError, ValueError, KeyError):
            event("embedding_cache_unreadable")
            return {}

    def _persist(self, model: str | None, embeddings: dict[str, np.ndarray]) -> None:
        path = self._cache_file(model)
        if path is None or not embeddings:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as destination:
                np.savez(destination, keys=np.array(list(embeddings)), vectors=np.vstack(list(embeddings.values())))
            os.replace(temporary, path)  # atomic, so a concurrent reader never sees a partial file
        except OSError:
            event("embedding_cache_write_failed")
