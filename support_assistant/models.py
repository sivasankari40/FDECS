from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping

from support_assistant.privacy import redact_text
from support_assistant.ticket_ids import canonical_ticket_id
from support_assistant.error_codes import extract_error_codes, normalize_error_codes


def required_text(data: Mapping[str, Any], field: str, limit: int) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string.")
    if len(value) > limit:
        raise ValueError(f"{field} must be at most {limit} characters.")
    return redact_text(value.strip())


@dataclass(frozen=True)
class TicketQuery:
    subject: str
    description: str
    error_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = {"subject": self.subject, "description": self.description}
        object.__setattr__(self, "subject", required_text(values, "subject", 300))
        object.__setattr__(self, "description", required_text(values, "description", 10000))
        object.__setattr__(self, "error_codes", normalize_error_codes(self.error_codes))


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    category: str
    subject: str
    description: str
    resolution_notes: str
    tags: tuple[str, ...]
    created_at: str
    status: str
    resolution_state: str = "approved"
    reviewed_at: str | None = None
    product_version: str | None = None
    legacy_ids: tuple[str, ...] = ()
    origin: str = "historical"
    revision: str | None = field(default=None, compare=False)
    error_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        original = self.ticket_id
        canonical = canonical_ticket_id(original)
        object.__setattr__(self, "ticket_id", canonical)
        aliases = set(self.legacy_ids)
        if original != canonical:
            aliases.add(original)
        if any(canonical_ticket_id(alias) != canonical for alias in aliases):
            raise ValueError("Legacy ticket aliases must map to the same canonical Ticket ID.")
        object.__setattr__(self, "legacy_ids", tuple(sorted(aliases)))

    @classmethod
    def incoming(cls, ticket_id: str, subject: str, description: str, category: str = "uncategorized",
                 error_codes: tuple[str, ...] | None = None) -> "Ticket":
        if category not in {"auth", "billing", "sync", "render", "perf", "api", "export", "perm", "security", "uncategorized"}:
            raise ValueError("Select a valid incoming ticket category.")
        document = {"ticket_id": ticket_id, "category": category, "subject": subject,
                              "description": description, "resolution_notes": "", "tags": [],
                              "created_at": datetime.now(timezone.utc).date().isoformat(), "status": "open",
                              "resolution_state": "needs_review", "reviewed_at": None, "origin": "incoming"}
        if error_codes is not None:
            document["error_codes"] = list(error_codes)
        return cls.from_dict(document)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Ticket":
        ticket_id = required_text(data, "ticket_id", 100)
        canonical_ticket_id(ticket_id)
        category = required_text(data, "category", 100)
        subject = required_text(data, "subject", 300)
        description = required_text(data, "description", 10000)
        status = required_text(data, "status", 100)
        if status not in {"open", "in_progress", "resolved", "closed"}:
            raise ValueError(f"{ticket_id}: status must be open, in_progress, resolved, or closed.")
        created_at = required_text(data, "created_at", 10)
        try:
            parsed_date = date.fromisoformat(created_at)
        except ValueError as error:
            raise ValueError(f"{ticket_id}: created_at must be a valid YYYY-MM-DD date.") from error
        if parsed_date.isoformat() != created_at:
            raise ValueError(f"{ticket_id}: created_at must use YYYY-MM-DD.")
        tags = data.get("tags")
        if not isinstance(tags, list) or len(tags) > 20 or any(
            not isinstance(tag, str) or not tag.strip() or len(tag) > 60 for tag in tags
        ):
            raise ValueError(f"{ticket_id}: tags must contain at most 20 nonempty strings of at most 60 characters.")
        resolution = data.get("resolution_notes")
        if not isinstance(resolution, str) or len(resolution) > 20000:
            raise ValueError(f"{ticket_id}: resolution_notes must be a string of at most 20000 characters.")
        if status == "resolved" and not resolution.strip():
            raise ValueError(f"{ticket_id}: resolved tickets need resolution notes.")
        state = data.get("resolution_state", "approved")
        if not isinstance(state, str) or state not in {"approved", "needs_review", "deprecated"}:
            raise ValueError(f"{ticket_id}: invalid resolution_state.")
        reviewed_at = data.get("reviewed_at")
        if reviewed_at is not None:
            if not isinstance(reviewed_at, str):
                raise ValueError(f"{ticket_id}: reviewed_at must be a YYYY-MM-DD date.")
            try:
                reviewed_date = date.fromisoformat(reviewed_at)
            except ValueError as error:
                raise ValueError(f"{ticket_id}: reviewed_at must be a valid date.") from error
            if reviewed_date.isoformat() != reviewed_at or reviewed_date < parsed_date:
                raise ValueError(f"{ticket_id}: reviewed_at must be YYYY-MM-DD and not precede creation.")
        version = data.get("product_version")
        if version is not None and (not isinstance(version, str) or not version.strip() or len(version) > 60):
            raise ValueError(f"{ticket_id}: product_version must be a nonempty string of at most 60 characters.")
        aliases = data.get("legacy_ids", [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ValueError("legacy_ids must be a list of ticket identifiers.")
        origin = data.get("origin", "historical")
        if not isinstance(origin, str) or origin not in {"historical", "incoming"}:
            raise ValueError("Ticket origin must be historical or incoming.")
        raw_error_codes = data.get("error_codes")
        if raw_error_codes is None:
            error_codes = extract_error_codes((subject, description, *tags))
        else:
            error_codes = normalize_error_codes(raw_error_codes)
        return cls(ticket_id, category, subject, description, redact_text(resolution.strip()),
                   tuple(redact_text(tag.strip()) for tag in tags), created_at, status,
                   state, reviewed_at, redact_text(version.strip()) if version else None, tuple(aliases), origin, data.get("_rev"), error_codes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "category": self.category,
            "subject": self.subject,
            "description": self.description,
            "resolution_notes": self.resolution_notes,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "status": self.status,
            "resolution_state": self.resolution_state,
            "reviewed_at": self.reviewed_at,
            "product_version": self.product_version,
            "legacy_ids": list(self.legacy_ids),
            "origin": self.origin,
            "error_codes": list(self.error_codes),
        }