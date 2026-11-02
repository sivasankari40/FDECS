import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from support_assistant.config import Settings
from support_assistant.matching import ABSENCE_REASONS
from support_assistant.observability import event
from support_assistant.repository import CouchRepository, RepositoryError
from support_assistant.ticket_ids import canonical_ticket_id


MANAGED_BY = "solari_support_assistant"
# Agents (non-admin identities) may only create or update well-formed votes and missed-precedent
# reports; nothing can be deleted and no free text is accepted.
FEEDBACK_VALIDATOR = """function(newDoc, oldDoc, userCtx) {
    if (userCtx.roles.indexOf('_admin') !== -1) { return; }
    if (newDoc._deleted) { throw({forbidden: 'Feedback cannot be deleted'}); }
    if (newDoc.type === 'missed_precedent') {
        if (!/^missed:[0-9a-f]{32}:T-[0-9]{6}$/.test(newDoc._id) ||
            newDoc._id !== 'missed:' + newDoc.search_id + ':' + newDoc.ticket_id ||
            (newDoc.incoming_ticket_id !== null && !/^T-[0-9]{6}$/.test(newDoc.incoming_ticket_id)) ||
            !Array.isArray(newDoc.shown_ticket_ids) || newDoc.shown_ticket_ids.length > 10 ||
            newDoc.shown_ticket_ids.some(function(id) { return !/^T-[0-9]{6}$/.test(id); }) ||
            ['not_indexed', 'outside_review_window', 'category_filter', 'error_code_conflict', 'below_threshold', 'ranked_below_shown'].indexOf(newDoc.reason) === -1 ||
            (newDoc.score !== null && typeof newDoc.score !== 'number') || typeof newDoc.reported_at !== 'string') {
            throw({forbidden: 'Invalid missed-precedent report'});
        }
        return;
    }
    if (newDoc.type !== 'feedback' || !/^feedback:[0-9a-f]{32}:T-[0-9]{6}$/.test(newDoc._id) ||
        newDoc._id !== 'feedback:' + newDoc.search_id + ':' + newDoc.ticket_id ||
        (newDoc.helpful !== true && newDoc.helpful !== false && newDoc.helpful !== null) ||
        (newDoc.incoming_ticket_id !== null && !/^T-[0-9]{6}$/.test(newDoc.incoming_ticket_id)) ||
        typeof newDoc.rank !== 'number' || typeof newDoc.score !== 'number' || typeof newDoc.voted_at !== 'string') {
        throw({forbidden: 'Invalid feedback document'});
    }
}"""
FEEDBACK_VIEWS = {
    "by_ticket": {
        "map": "function(doc) { if (doc.type === 'feedback' && doc.helpful !== null) { emit(doc.ticket_id, doc.helpful ? [1, 0] : [0, 1]); } }",
        "reduce": "_sum",
    },
}


def prepare_feedback_database(settings: Settings) -> None:
    """Create the feedback database, its write guard and vote view (administrator credentials)."""
    with CouchRepository(settings.for_feedback(admin=True)) as repository:
        repository.initialize()
        current = repository._request("GET", "/_design/feedback", allowed=(200, 404))
        if current.get("_rev") and current.get("managed_by") != MANAGED_BY:
            raise RepositoryError("Existing feedback design document is unmanaged; it was not changed.")
        design = {"_id": "_design/feedback", "managed_by": MANAGED_BY,
                  "validate_doc_update": FEEDBACK_VALIDATOR, "views": FEEDBACK_VIEWS}
        if (current.get("validate_doc_update"), current.get("views")) != (FEEDBACK_VALIDATOR, FEEDBACK_VIEWS):
            if current.get("_rev"):
                design["_rev"] = current["_rev"]
            repository._request("PUT", "/_design/feedback", allowed=(201, 202), json=design)
        if settings.reader_username:
            security = repository._request("GET", "/_security")
            members = security.setdefault("members", {"names": [], "roles": []})
            if settings.reader_username not in members.get("names", []):
                members["names"] = sorted(set(members.get("names", [])) | {settings.reader_username})
                repository._request("PUT", "/_security", json=security)


class FeedbackStore:
    """Persists agents' helpful/not-helpful votes. Stores IDs, rank and score only — never ticket text."""

    def __init__(self, settings: Settings) -> None:
        self.repository = CouchRepository(settings.for_feedback())

    def __enter__(self) -> "FeedbackStore":
        return self

    def __exit__(self, *args: Any) -> None:
        self.repository.__exit__(*args)

    def record(self, search_id: str, ticket_id: str, incoming_ticket_id: str | None, helpful: bool | None,
               rank: int, score: float) -> None:
        if not re.fullmatch(r"[0-9a-f]{32}", search_id):
            raise ValueError("Feedback requires a search ID.")
        document_id = f"feedback:{search_id}:{canonical_ticket_id(ticket_id)}"
        incoming = optional_ticket_id(incoming_ticket_id)
        path = "/" + quote(document_id, safe="")
        for _ in range(3):
            existing = self.repository._request("GET", path, allowed=(200, 404))
            document = {"_id": document_id, "type": "feedback", "search_id": search_id,
                        "ticket_id": canonical_ticket_id(ticket_id), "incoming_ticket_id": incoming,
                        "helpful": helpful, "rank": rank, "score": round(float(score), 4),
                        "voted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if existing.get("_rev"):
                document["_rev"] = existing["_rev"]
            if self.repository._request("PUT", path, allowed=(201, 202, 409), json=document).get("ok"):
                event("feedback_saved", helpful=helpful is True, cleared=helpful is None)
                return
        raise RepositoryError("Feedback changed concurrently and was not saved.")

    def counts(self, ticket_ids: Iterable[str]) -> dict[str, tuple[int, int]]:
        """Helpful and not-helpful vote totals for each precedent."""
        keys = list(dict.fromkeys(ticket_ids))
        if not keys:
            return {}
        result = self.repository._request("POST", "/_design/feedback/_view/by_ticket", params={"group": "true"},
                                          json={"keys": keys})
        return {row["key"]: (int(row["value"][0]), int(row["value"][1]))
                for row in result.get("rows", []) if isinstance(row, dict) and isinstance(row.get("value"), list)}

    def record_missed(self, search_id: str, ticket_id: str, incoming_ticket_id: str | None,
                      shown_ticket_ids: Iterable[str], reason: str, score: float | None) -> None:
        """An agent's report that a resolved precedent should have been suggested for this search."""
        if not re.fullmatch(r"[0-9a-f]{32}", search_id):
            raise ValueError("Missed-precedent reports require a search ID.")
        if reason not in ABSENCE_REASONS or reason == "shown":
            raise ValueError("Unknown missed-precedent reason.")
        precedent = canonical_ticket_id(ticket_id)
        document_id = f"missed:{search_id}:{precedent}"
        path = "/" + quote(document_id, safe="")
        existing = self.repository._request("GET", path, allowed=(200, 404))
        document = {"_id": document_id, "type": "missed_precedent", "search_id": search_id, "ticket_id": precedent,
                    "incoming_ticket_id": optional_ticket_id(incoming_ticket_id),
                    "shown_ticket_ids": [canonical_ticket_id(shown) for shown in shown_ticket_ids][:10],
                    "reason": reason, "score": None if score is None else round(float(score), 4),
                    "reported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if existing.get("_rev"):
            document["_rev"] = existing["_rev"]
        if not self.repository._request("PUT", path, allowed=(201, 202, 409), json=document).get("ok"):
            raise RepositoryError("Missed-precedent report changed concurrently and was not saved.")
        event("missed_precedent_reported")

    def votes(self) -> list[dict[str, Any]]:
        return [vote for vote in self._documents("feedback:") if vote.get("helpful") is not None]

    def missed(self) -> list[dict[str, Any]]:
        return self._documents("missed:")

    def _documents(self, prefix: str) -> list[dict[str, Any]]:
        documents, start_key = [], prefix
        while True:
            result = self.repository._request("GET", "/_all_docs", params={
                "include_docs": "true", "limit": 501, "startkey": json.dumps(start_key), "endkey": json.dumps(prefix + "\uffff")})
            rows = result.get("rows", [])
            documents.extend(row["doc"] for row in rows[:500] if isinstance(row.get("doc"), dict))
            if len(rows) <= 500:
                return documents
            start_key = rows[500]["key"]


def optional_ticket_id(value: str | None) -> str | None:
    try:
        return canonical_ticket_id(value) if value else None
    except ValueError:
        return None


SCORE_BANDS = ((0.0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.80), (0.80, 1.01))


def rate(helpful: int, total: int) -> float | None:
    return round(helpful / total, 3) if total else None


def feedback_report(votes: list[dict[str, Any]], missed: list[dict[str, Any]], min_votes: int = 3) -> dict[str, Any]:
    """Summarize agents' votes and missed-precedent reports into actions for support leads."""
    bands = []
    for low, high in SCORE_BANDS:
        in_band = [vote for vote in votes if low <= vote["score"] < high]
        helpful = sum(vote["helpful"] for vote in in_band)
        bands.append({"score_band": f"{low:.2f}-{min(high, 1.0):.2f}", "votes": len(in_band), "helpful_rate": rate(helpful, len(in_band))})
    per_precedent: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for vote in votes:
        per_precedent[vote["ticket_id"]][0 if vote["helpful"] else 1] += 1
    review = sorted(({"ticket_id": ticket_id, "helpful": helpful, "not_helpful": unhelpful, "helpful_rate": rate(helpful, helpful + unhelpful)}
                     for ticket_id, (helpful, unhelpful) in per_precedent.items()
                     if helpful + unhelpful >= min_votes and helpful / (helpful + unhelpful) < 0.5),
                    key=lambda row: (row["helpful_rate"], -row["not_helpful"], row["ticket_id"]))
    weekly: dict[str, dict[str, int]] = defaultdict(lambda: {"votes": 0, "helpful": 0, "missed": 0})
    for vote in votes:
        week = datetime.fromisoformat(vote["voted_at"]).strftime("%G-W%V")
        weekly[week]["votes"] += 1
        weekly[week]["helpful"] += bool(vote["helpful"])
    for report in missed:
        weekly[datetime.fromisoformat(report["reported_at"]).strftime("%G-W%V")]["missed"] += 1
    helpful_total = sum(vote["helpful"] for vote in votes)
    return {
        "votes": len(votes),
        "helpful_rate": rate(helpful_total, len(votes)),
        "searches_with_votes": len({vote["search_id"] for vote in votes}),
        "by_score_band": bands,
        "precedents_to_review": review,
        "missed_precedents": {
            "reports": len(missed),
            "by_reason": dict(Counter(report["reason"] for report in missed).most_common()),
            "most_reported": [{"ticket_id": ticket_id, "reports": count}
                              for ticket_id, count in Counter(report["ticket_id"] for report in missed).most_common(10)],
        },
        "weekly": [{"week": week, "votes": row["votes"], "helpful_rate": rate(row["helpful"], row["votes"]), "missed_reports": row["missed"]}
                   for week, row in sorted(weekly.items())],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize or export agents' votes and missed-precedent reports.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--report", action="store_true", help="Print helpful rate by score band, precedents to review, missed precedents, and weekly trend.")
    action.add_argument("--export", type=Path, help="Write the votes as labelled pairs to this JSON file.")
    parser.add_argument("--min-votes", type=int, default=3, help="Votes a precedent needs before it can be flagged for review.")
    args = parser.parse_args()
    try:
        with FeedbackStore(Settings.from_env(role="reader")) as store:
            votes = store.votes()
            missed = store.missed()
        if args.report:
            print(json.dumps(feedback_report(votes, missed, args.min_votes), indent=2))
            return 0
        labels = [{key: vote[key] for key in ("incoming_ticket_id", "ticket_id", "helpful", "rank", "score", "voted_at")}
                  for vote in sorted(votes, key=lambda vote: vote["voted_at"])]
        missed_labels = [{key: report[key] for key in ("incoming_ticket_id", "ticket_id", "reason", "score", "reported_at")} for report in missed]
        args.export.write_text(json.dumps({"synthetic": False, "labels": labels, "missed_precedents": missed_labels}, indent=2), encoding="utf-8")
        helpful = sum(label["helpful"] for label in labels)
        print(f"Exported {len(labels)} votes ({helpful} helpful, {len(labels) - helpful} not helpful) and {len(missed_labels)} missed-precedent reports to {args.export}")
        return 0
    except (ValueError, RepositoryError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
