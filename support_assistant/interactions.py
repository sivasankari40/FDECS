from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from support_assistant.models import TicketQuery


CATEGORY_FILTERS = {
    "Auth": frozenset({"auth"}),
    "Billing": frozenset({"billing"}),
    "Sync": frozenset({"sync"}),
    "Dashboard": frozenset({"render", "perf"}),
    "API": frozenset({"api"}),
    "Export": frozenset({"export"}),
    "Permissions": frozenset({"perm"}),
    "Security": frozenset({"security"}),
}


def category_codes(labels: tuple[str, ...] | list[str]) -> frozenset[str] | None:
    if any(label not in CATEGORY_FILTERS for label in labels):
        raise ValueError("Unknown category filter.")
    if not labels:
        return None
    return frozenset(code for label in labels for code in CATEGORY_FILTERS[label])


@dataclass
class SearchEntry:
    search_id: str
    query: TicketQuery
    categories: tuple[str, ...]
    searched_at: str
    result_ids: tuple[str, ...]
    fingerprint: str
    votes: dict[str, int] = field(default_factory=dict)
    incoming_ticket_id: str | None = None


class InteractionHistory:
    def __init__(self, limit: int = 20) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("Search history limit must be between 1 and 100.")
        self.entries: deque[SearchEntry] = deque(maxlen=limit)

    def record(self, query: TicketQuery, categories: list[str], result_ids: tuple[str, ...], fingerprint: str,
               incoming_ticket_id: str | None = None) -> SearchEntry:
        category_codes(categories)
        entry = SearchEntry(uuid4().hex, query, tuple(categories), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            result_ids, fingerprint, incoming_ticket_id=incoming_ticket_id)
        self.entries.appendleft(entry)
        return entry

    def find(self, search_id: str) -> SearchEntry:
        entry = next((entry for entry in self.entries if entry.search_id == search_id), None)
        if entry is None:
            raise ValueError("That search is no longer in this session's history.")
        return entry

    def vote(self, search_id: str, ticket_id: str, value: int | None) -> None:
        entry = self.find(search_id)
        if ticket_id not in entry.result_ids:
            raise ValueError("Feedback must refer to a result of the selected search.")
        if value is None:
            entry.votes.pop(ticket_id, None)
        elif isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
            raise ValueError("Feedback must be helpful, not helpful, or cleared.")
        else:
            entry.votes[ticket_id] = value

    def clear(self) -> None:
        self.entries.clear()