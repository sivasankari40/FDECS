import argparse
import json
import sys
from collections import Counter, defaultdict
from typing import Any, Callable

from support_assistant.config import Settings
from support_assistant.feedback import FeedbackStore
from support_assistant.matching import SEMANTIC_SCORE_WEIGHT, SimilarityIndex
from support_assistant.models import Ticket, TicketQuery
from support_assistant.repository import CouchRepository, RepositoryError


THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
SEMANTIC_WEIGHTS = (0.35, 0.45, 0.55)


def feedback_cases(votes: list[dict[str, Any]], missed: list[dict[str, Any]],
                   lookup: Callable[[str], Ticket | None]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Turn agents' votes and missed-precedent reports into evaluation cases, one per incoming ticket.

    A precedent is relevant when most of its votes were helpful or an agent reported it as missed,
    irrelevant when most votes were not helpful; ties are dropped. The incoming ticket's text is
    fetched by ID, because feedback stores no ticket text.
    """
    judgements: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    skipped: Counter = Counter()
    for vote in votes:
        if vote.get("incoming_ticket_id"):
            judgements[vote["incoming_ticket_id"]][vote["ticket_id"]].append(bool(vote["helpful"]))
        else:
            skipped["votes_without_incoming_ticket"] += 1
    for report in missed:
        if report.get("incoming_ticket_id"):
            judgements[report["incoming_ticket_id"]][report["ticket_id"]].append(True)
        else:
            skipped["missed_reports_without_incoming_ticket"] += 1
    cases = []
    for incoming_id in sorted(judgements):
        ticket = lookup(incoming_id)
        if ticket is None:
            skipped["incoming_ticket_no_longer_available"] += 1
            continue
        relevant = sorted(precedent for precedent, values in judgements[incoming_id].items() if sum(values) > len(values) / 2)
        irrelevant = sorted(precedent for precedent, values in judgements[incoming_id].items() if sum(values) < len(values) / 2)
        if not relevant and not irrelevant:
            skipped["tied_judgements"] += 1
            continue
        cases.append({"incoming_ticket_id": incoming_id, "subject": ticket.subject, "description": ticket.description,
                      "error_codes": list(ticket.error_codes), "relevant": relevant, "irrelevant": irrelevant})
    return cases, dict(skipped)


def evaluate_feedback(index: SimilarityIndex, cases: list[dict[str, Any]], min_score: float,
                      max_age_days: int | None = None) -> dict[str, Any]:
    found = relevant_total = unhelpful_returned = unhelpful_total = unjudged = unreachable = 0
    cases_with_helpful_result = 0
    eligible = index.eligible_mask(max_age_days)
    for case in cases:
        query = TicketQuery(case["subject"], case["description"], tuple(case["error_codes"]))
        returned = {match.ticket.ticket_id for match in index.search(query, min_score=min_score, exclude_ticket_id=case["incoming_ticket_id"],
                                                                     max_age_days=max_age_days)}
        relevant, irrelevant = set(case["relevant"]), set(case["irrelevant"])
        # Relevant precedents that are not approved or outside the review window cannot be fixed by tuning.
        unreachable += sum(1 for ticket_id in relevant if ticket_id not in index.positions or not eligible[index.positions[ticket_id]])
        found += len(relevant & returned)
        relevant_total += len(relevant)
        unhelpful_returned += len(irrelevant & returned)
        unhelpful_total += len(irrelevant)
        unjudged += len(returned - relevant - irrelevant)
        cases_with_helpful_result += bool(relevant & returned)
    return {
        "threshold": min_score,
        "semantic_weight": index.semantic_weight if index.semantic_matrix is not None else None,
        "helpful_recall": round(found / relevant_total, 3) if relevant_total else None,
        "unhelpful_shown_rate": round(unhelpful_returned / unhelpful_total, 3) if unhelpful_total else None,
        "cases_with_a_helpful_result": cases_with_helpful_result,
        "unjudged_results": unjudged,
        "unreachable_relevant": unreachable,
    }


def sweep(index: SimilarityIndex, cases: list[dict[str, Any]], max_age_days: int | None = None,
          thresholds: tuple[float, ...] = THRESHOLDS) -> list[dict[str, Any]]:
    weights = SEMANTIC_WEIGHTS if index.semantic_matrix is not None else (index.semantic_weight,)
    original = index.semantic_weight
    try:
        rows = []
        for weight in weights:
            index.semantic_weight = weight
            rows.extend(evaluate_feedback(index, cases, threshold, max_age_days) for threshold in thresholds)
        return rows
    finally:
        index.semantic_weight = original


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay agents' feedback against the current history and compare ranking settings.")
    parser.add_argument("--max-age-days", type=int, default=None, help="Defaults to RESOLUTION_MAX_AGE_DAYS.")
    args = parser.parse_args()
    try:
        settings = Settings.from_env(role="reader")
        max_age = settings.resolution_age_days if args.max_age_days is None else args.max_age_days
        with FeedbackStore(settings) as store:
            votes, missed = store.votes(), store.missed()
        with CouchRepository(settings.for_reader()) as history, CouchRepository(settings.for_incoming()) as incoming:
            index = SimilarityIndex(history.list_tickets())
            cases, skipped = feedback_cases(votes, missed, incoming.get_ticket)
        current = {"threshold": settings.min_score, "semantic_weight": SEMANTIC_SCORE_WEIGHT if index.semantic_matrix is not None else None}
        print(json.dumps({
            "cases": len(cases),
            "skipped": skipped,
            "current_settings": {**current, **evaluate_feedback(index, cases, settings.min_score, max_age)} if cases else None,
            "sweep": sweep(index, cases, max_age) if cases else [],
            "how_to_read": ("Prefer settings that keep helpful_recall high while lowering unhelpful_shown_rate. "
                            "unjudged_results are precedents agents never rated: a lower threshold adds them, so "
                            "collect more votes before trusting a lower threshold. unreachable_relevant precedents are "
                            "unapproved or outside the review window; fix those in the knowledge base, not by tuning."),
        }, indent=2))
        return 0
    except (ValueError, RepositoryError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
