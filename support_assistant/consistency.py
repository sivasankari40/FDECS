import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date

import numpy as np

from support_assistant.config import ROOT, Settings
from support_assistant.matching import SimilarityIndex
from support_assistant.models import Ticket
from support_assistant.repository import CouchRepository, RepositoryError, load_history


# Issue similarity between two precedents is semantic cosine when the model is loaded, otherwise TF-IDF cosine,
# which runs on a much lower scale. Calibrated on the synthetic history: the closest look-alike pair with a
# different fix (duplicate charge vs pending authorization) scores 0.68 semantic / 0.18 TF-IDF.
WARN_ISSUE_SIMILARITY = {"semantic": 0.60, "lexical": 0.25}
# The report targets the same problem answered differently, so it requires near-duplicate issues.
REPORT_ISSUE_SIMILARITY = {"semantic": 0.75, "lexical": 0.50}
# Resolution notes below this TF-IDF cosine are treated as different answers.
DIFFERENT_RESOLUTION = 0.20


@dataclass(frozen=True)
class ResolutionConflict:
    first: Ticket
    second: Ticket
    issue_similarity: float
    resolution_similarity: float


def similarity_mode(index: SimilarityIndex) -> str:
    return "semantic" if index.semantic_matrix is not None else "lexical"


def issue_vectors(index: SimilarityIndex):
    return index.semantic_matrix if index.semantic_matrix is not None else index.matrix


def pair_similarity(matrix, first: int, second: int) -> float:
    value = matrix[first] @ matrix[second].T
    return float(value.toarray()[0, 0] if hasattr(value, "toarray") else value)


def conflicts_among(index: SimilarityIndex, tickets: list[Ticket] | tuple[Ticket, ...],
                    issue_threshold: float | None = None) -> tuple[ResolutionConflict, ...]:
    """Pairs of precedents that describe similar problems but were resolved differently."""
    vectors = issue_vectors(index)
    if vectors is None or index.resolution_matrix is None:
        return ()
    threshold = WARN_ISSUE_SIMILARITY[similarity_mode(index)] if issue_threshold is None else issue_threshold
    positions = [index.positions[ticket.ticket_id] for ticket in tickets if ticket.ticket_id in index.positions]
    conflicts = []
    for offset, first in enumerate(positions):
        for second in positions[offset + 1:]:
            issue = pair_similarity(vectors, first, second)
            resolution = pair_similarity(index.resolution_matrix, first, second)
            if issue >= threshold and resolution < DIFFERENT_RESOLUTION:
                conflicts.append(ResolutionConflict(index.indexed[first], index.indexed[second], issue, resolution))
    return tuple(sorted(conflicts, key=lambda conflict: -conflict.issue_similarity))


def conflict_report(index: SimilarityIndex, max_age_days: int | None = None, today: date | None = None,
                    chunk: int = 256) -> tuple[ResolutionConflict, ...]:
    """All eligible near-duplicate issues with different resolutions, for support leads to reconcile."""
    vectors = issue_vectors(index)
    if vectors is None or index.resolution_matrix is None:
        return ()
    threshold = REPORT_ISSUE_SIMILARITY[similarity_mode(index)]
    eligible = np.flatnonzero(index.eligible_mask(max_age_days, today))
    conflicts = []
    # Chunked so memory stays bounded (chunk x corpus similarities) at the 50,000-ticket cap.
    for start in range(0, len(eligible), chunk):
        rows = eligible[start:start + chunk]
        similarities = vectors[rows] @ vectors[eligible].T
        similarities = similarities.toarray() if hasattr(similarities, "toarray") else similarities
        for row, column in zip(*np.nonzero(similarities >= threshold)):
            first, second = int(rows[row]), int(eligible[column])
            if first >= second:
                continue
            resolution = pair_similarity(index.resolution_matrix, first, second)
            if resolution < DIFFERENT_RESOLUTION:
                conflicts.append(ResolutionConflict(index.indexed[first], index.indexed[second],
                                                    float(similarities[row, column]), resolution))
    return tuple(sorted(conflicts, key=lambda conflict: -conflict.issue_similarity))


def main() -> int:
    parser = argparse.ArgumentParser(description="Report similar resolved tickets whose resolutions differ, for support leads to reconcile.")
    parser.add_argument("--source", choices=("couchdb", "files"), default="couchdb",
                        help="Read history from CouchDB (default) or the bundled synthetic JSON files.")
    parser.add_argument("--max-age-days", type=int, default=None, help="Defaults to RESOLUTION_MAX_AGE_DAYS.")
    args = parser.parse_args()
    try:
        settings = Settings.from_env(role="reader")
        if args.source == "files":
            history = load_history(ROOT / "starter_kit" / "data" / "tickets.json") + load_history(ROOT / "data" / "production_tickets.json")
        else:
            with CouchRepository(settings.for_reader()) as repository:
                history = repository.list_tickets()
        index = SimilarityIndex(history)
        max_age = settings.resolution_age_days if args.max_age_days is None else args.max_age_days
        conflicts = conflict_report(index, max_age)
        print(json.dumps({
            "mode": similarity_mode(index),
            "issue_similarity_threshold": REPORT_ISSUE_SIMILARITY[similarity_mode(index)],
            "eligible_tickets": int(index.eligible_mask(max_age).sum()),
            "conflicts": [{"tickets": [conflict.first.ticket_id, conflict.second.ticket_id],
                           "subjects": [conflict.first.subject, conflict.second.subject],
                           "issue_similarity": round(conflict.issue_similarity, 3),
                           "resolution_similarity": round(conflict.resolution_similarity, 3)} for conflict in conflicts],
        }, indent=2))
        return 0
    except (ValueError, RepositoryError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
