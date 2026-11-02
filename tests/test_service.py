from pathlib import Path
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest

from support_assistant.repository import RepositoryError, load_history
from support_assistant.config import Settings
from support_assistant.matching import search_text
from support_assistant.service import IndexStore, make_snapshot, read_incremental_snapshot, read_snapshot


DATA = Path(__file__).resolve().parents[1] / "starter_kit" / "data" / "tickets.json"


def test_snapshot_retries_concurrent_change():
    repository = Mock()
    repository.info.side_effect = [{"update_seq": sequence} for sequence in [1, 2, 2, 2]]
    repository.list_tickets.return_value = load_history(DATA)
    snapshot = read_snapshot(repository)
    assert snapshot.sequence == 2
    assert len(snapshot.tickets) == 40
    assert repository.list_tickets.call_count == 2


def test_snapshot_fingerprint_tracks_content():
    repository = Mock()
    repository.info.return_value = {"update_seq": 1}
    repository.list_tickets.return_value = load_history(DATA)
    first = read_snapshot(repository)
    assert first.fingerprint == read_snapshot(repository).fingerprint
    repository.list_tickets.return_value = load_history(DATA)[1:]
    assert first.fingerprint != read_snapshot(repository).fingerprint


def test_continuous_changes_fail_explicitly():
    repository = Mock()
    repository.info.side_effect = [{"update_seq": sequence} for sequence in range(6)]
    repository.list_tickets.return_value = []
    with pytest.raises(RepositoryError, match="changed during refresh"):
        read_snapshot(repository)


def test_incremental_update_and_deletion_without_full_read():
    history = load_history(DATA)
    repository = Mock(settings=Settings())
    updated = replace(history[0], subject="Updated password reset incident")
    repository.changes.return_value = {"results": [
        {"id": "ticket:" + updated.ticket_id, "doc": {"_id": "ticket:" + updated.ticket_id, "type": "ticket", **updated.to_dict()}},
        {"id": "ticket:" + history[1].ticket_id, "deleted": True},
        {"id": "_design/read_access", "doc": {"type": "design"}},
    ], "last_seq": 2, "pending": 0}
    repository.info.return_value = {"update_seq": 2}
    snapshot = read_incremental_snapshot(repository, make_snapshot(tuple(history), 1))
    assert len(snapshot.tickets) == 39
    assert snapshot.tickets[0].subject == updated.subject
    repository.list_tickets.assert_not_called()


def test_failed_incremental_refresh_does_not_publish_partial_state():
    history = load_history(DATA)
    repository = Mock(settings=Settings())
    repository.info.return_value = {"update_seq": 2}
    repository.changes.return_value = {"results": [{"id": "ticket:T-001001", "doc": {"_id": "ticket:T-001001", "type": "ticket"}}], "last_seq": 2, "pending": 0}
    store = IndexStore()
    store.snapshot = make_snapshot(tuple(history), 1)
    previous = store.snapshot
    with pytest.raises(RepositoryError, match="invalid"):
        store.refresh(repository)
    assert store.snapshot is previous


def test_shared_store_reuses_unchanged_history_and_index():
    history = load_history(DATA)
    repository = Mock(settings=Settings())
    repository.info.return_value = {"update_seq": 1}
    repository.list_tickets.return_value = history
    store = IndexStore()
    first = store.refresh(repository)
    index = store.index_for(first)
    assert store.refresh(repository) is first
    assert store.index_for(first) is index
    assert repository.list_tickets.call_count == 1


def test_changes_feed_must_advance():
    repository = Mock(settings=Settings())
    repository.info.return_value = {"update_seq": 1}
    repository.changes.return_value = {"results": [], "last_seq": 1, "pending": 1}
    with pytest.raises(RepositoryError, match="did not advance"):
        read_incremental_snapshot(repository, make_snapshot((), 1))


def test_database_and_feed_tokens_are_not_compared_across_apis():
    repository = Mock(settings=Settings())
    repository.info.return_value = {"update_seq": "opaque-info-version-2"}
    repository.changes.return_value = {"results": [], "last_seq": "opaque-feed-cursor-2", "pending": 0}
    snapshot = read_incremental_snapshot(repository, make_snapshot((), "opaque-info-version-1", "opaque-feed-cursor-1"))
    assert snapshot.sequence == "opaque-info-version-2"
    assert snapshot.change_sequence == "opaque-feed-cursor-2"
    repository.changes.assert_called_once_with("opaque-feed-cursor-1")

def test_index_rebuild_only_encodes_changed_tickets(monkeypatch):
    class Encoder:
        def __init__(self):
            self.encoded = []

        def encode(self, texts, **kwargs):
            self.encoded.extend(texts)
            return np.ones((len(texts), 2))

    encoder = Encoder()
    monkeypatch.setattr("support_assistant.matching.semantic_model_source", lambda: "test-model")
    monkeypatch.setattr("support_assistant.matching.load_sentence_model", lambda source: encoder)
    history = load_history(DATA)
    store = IndexStore()
    store.index_for(make_snapshot(tuple(history), 1))
    assert len(encoder.encoded) == len(history)
    edited = replace(history[0], subject="Edited password reset incident")
    encoder.encoded.clear()
    index = store.index_for(make_snapshot((edited, *history[1:]), 2))
    assert encoder.encoded == [search_text(edited.subject, edited.description)]
    assert index.semantic_matrix.shape == (len(history), 2)
    assert len(store.embeddings) == len(history)


class CountingEncoder:
    def __init__(self):
        self.encoded = []

    def encode(self, texts, **kwargs):
        self.encoded.extend(texts)
        return np.ones((len(texts), 2))


@pytest.fixture
def counting_encoder(monkeypatch):
    encoder = CountingEncoder()
    monkeypatch.setattr("support_assistant.matching.semantic_model_source", lambda: "test-model")
    monkeypatch.setattr("support_assistant.service.semantic_model_source", lambda: "test-model")
    monkeypatch.setattr("support_assistant.matching.load_sentence_model", lambda source: encoder)
    return encoder


def test_embeddings_persist_across_restarts_without_ticket_text(counting_encoder, tmp_path):
    history = load_history(DATA)
    IndexStore(cache_directory=tmp_path).index_for(make_snapshot(tuple(history), 1))
    assert len(counting_encoder.encoded) == len(history)
    cache_files = list(tmp_path.glob("*.npz"))
    assert len(cache_files) == 1
    assert history[0].subject.encode("utf-8") not in cache_files[0].read_bytes()
    counting_encoder.encoded.clear()
    restarted = IndexStore(cache_directory=tmp_path).index_for(make_snapshot(tuple(history), 1))
    assert counting_encoder.encoded == []  # a cold start reuses every stored embedding
    assert restarted.semantic_matrix.shape == (len(history), 2)


def test_background_rebuild_serves_previous_index_until_ready(counting_encoder):
    history = load_history(DATA)
    store = IndexStore(background=True)
    first = store.index_for(make_snapshot(tuple(history), 1))
    changed = make_snapshot(tuple(history[1:]), 2)
    assert store.index_for(changed) is first  # the agent is not blocked by the rebuild
    store.builder.join(timeout=30)
    assert not store.rebuilding
    rebuilt = store.index_for(changed)
    assert rebuilt is not first and rebuilt.fingerprint == changed.fingerprint
    assert len(rebuilt.indexed) == len(history) - 1


def test_failed_background_rebuild_keeps_previous_index(counting_encoder, monkeypatch):
    history = load_history(DATA)
    store = IndexStore(background=True)
    first = store.index_for(make_snapshot(tuple(history), 1))

    def broken(*args, **kwargs):
        raise MemoryError("simulated")

    monkeypatch.setattr("support_assistant.service.SimilarityIndex", broken)
    changed = make_snapshot(tuple(history[1:]), 2)
    assert store.index_for(changed) is first
    store.builder.join(timeout=30)
    assert store.index is first and not store.rebuilding
