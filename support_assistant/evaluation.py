import argparse
import json
import sys
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np

from support_assistant.config import ROOT
from support_assistant.matching import DEFAULT_MIN_SCORE, SimilarityIndex
from support_assistant.models import TicketQuery
from support_assistant.ticket_ids import canonical_ticket_id
from support_assistant.repository import load_history


CORPUS_AS_OF = date(2026, 10, 31)

def load_cases(path: Path = ROOT / "data" / "evaluation_cases.json") -> list[dict]:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if payload.get("synthetic") is not True or not isinstance(payload.get("cases"), list):
        raise ValueError("Evaluation requires explicitly synthetic cases.")
    seen = set()
    for case in payload["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or case["case_id"] in seen:
            raise ValueError("Evaluation case IDs must be unique strings.")
        seen.add(case["case_id"])
        TicketQuery(case.get("subject"), case.get("description"), tuple(case.get("error_codes", [])))
        relevant = case.get("relevant_ticket_ids")
        if case.get("split") not in {"development", "heldout"} or not isinstance(relevant, list) or any(not isinstance(value, str) for value in relevant):
            raise ValueError("Evaluation split and relevant ticket IDs must be valid.")
        if not isinstance(case.get("should_abstain"), bool) or case["should_abstain"] != (len(relevant) == 0):
            raise ValueError("Abstention labels must agree with relevant ticket IDs.")
        case["relevant_ticket_ids"] = [canonical_ticket_id(ticket_id) for ticket_id in relevant]
    return payload["cases"]


def evaluate(index: SimilarityIndex, cases: list[dict], min_score: float = DEFAULT_MIN_SCORE) -> dict:
    outcomes = []
    elapsed = []
    for case in cases:
        query = TicketQuery(case["subject"], case["description"], tuple(case.get("error_codes", [])))
        start = perf_counter()
        matches = index.search(query, min_score=min_score)
        elapsed.append((perf_counter() - start) * 1000)
        returned = [match.ticket.ticket_id for match in matches]
        relevant = set(case["relevant_ticket_ids"])
        outcomes.append({"case_id": case["case_id"], "should_abstain": case["should_abstain"],
                         "returned_ids": returned, "recall_at_3": len(relevant.intersection(returned)) / len(relevant) if relevant else None,
                         "passed": not returned if case["should_abstain"] else bool(relevant.intersection(returned)),
                         "irrelevant_results": len(set(returned) - relevant)})
    positives = [outcome for outcome in outcomes if not outcome["should_abstain"]]
    negatives = [outcome for outcome in outcomes if outcome["should_abstain"]]
    return {"synthetic_only": True,
            "retrieval_mode": "hybrid" if index.semantic_matrix is not None else "lexical_fallback",
            "semantic_search_enabled": index.semantic_matrix is not None,
            "case_count": len(outcomes), "eligible_history": len(index.tickets),
            "threshold": min_score, "mean_recall_at_3": float(np.mean([outcome["recall_at_3"] for outcome in positives])) if positives else None,
            "correct_abstention_rate": sum(outcome["passed"] for outcome in negatives) / len(negatives) if negatives else None,
            "no_match_false_suggestion_rate": sum(not outcome["passed"] for outcome in negatives) / len(negatives) if negatives else None,
            "irrelevant_returned_count": sum(outcome["irrelevant_results"] for outcome in outcomes),
            "matching_p95_ms": float(np.percentile(elapsed, 95)) if elapsed else None,
            "outcomes": outcomes}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate synthetic ticket retrieval without CouchDB or remote inference.")
    parser.add_argument("--split", choices=("development", "heldout"), default="heldout")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--max-age-days", type=int, default=180)
    parser.add_argument("--as-of", type=date.fromisoformat, default=CORPUS_AS_OF,
                        help="Review-age reference date (YYYY-MM-DD); fixed by default so results do not drift with the clock.")
    args = parser.parse_args()
    try:
        history = load_history(ROOT / "starter_kit" / "data" / "tickets.json") + load_history(ROOT / "data" / "production_tickets.json")
        cases = [case for case in load_cases() if case["split"] == args.split]
        index = SimilarityIndex(history, max_age_days=args.max_age_days, today=args.as_of)
        report = evaluate(index, cases, args.min_score)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())