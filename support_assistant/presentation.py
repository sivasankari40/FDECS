from dataclasses import dataclass
from functools import lru_cache
from html import escape
import json
from pathlib import Path
import re

from support_assistant.interactions import CATEGORY_FILTERS
from support_assistant.matching import Match
from support_assistant.models import TicketQuery


def term_badges(terms: tuple[str, ...]) -> str:
    chips = []
    for term in terms:
        label = term[:48] + ("..." if len(term) > 48 else "")
        chips.append(f'<span class="term-chip" title="{escape(term, quote=True)}">{escape(label)}</span>')
    return '<div class="term-list">' + "".join(chips) + "</div>"


def match_reasons(match: Match, filters: tuple[str, ...], threshold: float) -> tuple[str, ...]:
    group = next((label for label, codes in CATEGORY_FILTERS.items() if match.ticket.category in codes), match.ticket.category)
    reasons = []
    if match.matched_terms:
        reasons.append(f"Shared subject/description terms: {', '.join(match.matched_terms)}.")
    if match.matched_tags:
        reasons.append(f"Matching historical tags: {', '.join(match.matched_tags)}.")
    reasons.extend((f"Historical category: {group}.",
                    f"Combined evidence score {match.score:.1%} meets the {threshold:.1%} recommendation threshold.",
                    f"Subject/description text similarity: {match.text_score:.1%}."))
    if match.semantic_score is not None:
        reasons.append(f"Sentence Transformer semantic similarity: {match.semantic_score:.1%}.")
    if match.matched_error_codes:
        reasons.append(f"Exact error-code match: {', '.join(match.matched_error_codes)} (strong matching evidence; still verify root cause).")
    selected = [label for label in filters if match.ticket.category in CATEGORY_FILTERS.get(label, ())]
    if selected:
        reasons.append(f"Within your selected category filter: {', '.join(selected)}.")
    reasons.append("Text and semantic similarity do not establish the same root cause or guarantee that this resolution applies.")
    return tuple(reasons)


@dataclass(frozen=True)
class TriageAdvice:
    team: str
    actions: tuple[str, ...]


@dataclass(frozen=True)
class TriageRouting:
    default_team: str
    routes: tuple[tuple[str, frozenset[str]], ...]


@lru_cache(maxsize=4)
def load_triage_routing(path: Path) -> TriageRouting:
    """Load keyword-to-team routing rules; the first route with a matching keyword wins."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Triage routing file {Path(path).name} must be readable JSON.") from error
    default_team = payload.get("default_team") if isinstance(payload, dict) else None
    routes = payload.get("routes") if isinstance(payload, dict) else None
    if not isinstance(default_team, str) or not default_team.strip() or not isinstance(routes, list):
        raise ValueError("Triage routing needs a default_team and a routes list.")
    parsed = []
    for route in routes:
        team = route.get("team") if isinstance(route, dict) else None
        keywords = route.get("keywords") if isinstance(route, dict) else None
        if (not isinstance(team, str) or not team.strip() or not isinstance(keywords, list) or not keywords
                or any(not isinstance(keyword, str) or not re.fullmatch(r"\w+", keyword) for keyword in keywords)):
            raise ValueError("Each triage route needs a team and a nonempty list of single-word keywords.")
        parsed.append((team.strip(), frozenset(keyword.casefold() for keyword in keywords)))
    return TriageRouting(default_team.strip(), tuple(parsed))


def no_match_advice(query: TicketQuery, filters: tuple[str, ...], routing: TriageRouting) -> TriageAdvice:
    tokens = set(re.findall(r"\w+", (query.subject + " " + query.description).casefold()))
    team = next((team for team, keywords in routing.routes if tokens.intersection(keywords)), routing.default_team)
    actions = [f"Escalate to {team} with the exact error, affected feature, and reproduction steps."]
    if filters:
        actions.insert(0, "Review the category filters; relevant precedents may be outside the current selection.")
    actions.append("After resolution, submit a knowledge-base article for review before reuse.")
    return TriageAdvice(team, tuple(actions))


def absence_explanation(reason: str, score: float | None, threshold: float) -> str:
    """Plain-language reason a precedent the agent expected was not shown."""
    explanations = {
        "not_indexed": "it is not an approved resolved ticket, so it is never suggested.",
        "outside_review_window": "its resolution review is outside the review window or has no review date.",
        "category_filter": "it is outside the selected category filter.",
        "error_code_conflict": "its recorded error code conflicts with the code entered for this ticket.",
        "below_threshold": f"its similarity score ({score or 0:.0%}) is below the {threshold:.0%} threshold.",
        "ranked_below_shown": f"it scored {score or 0:.0%}, but other precedents ranked higher.",
    }
    return explanations[reason]


def resolution_draft(match: Match) -> str:
    return f"Historical precedent: {match.ticket.ticket_id} - {match.ticket.subject}\n\n{match.ticket.resolution_notes}"