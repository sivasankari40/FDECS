import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from support_assistant.models import Ticket, TicketQuery
from support_assistant.semantic import SemanticModelUnavailable, encode_sentences, load_sentence_model, semantic_model_source


DEFAULT_MIN_SCORE = 0.30
# MiniLM scores even unrelated support tickets ~0.3 cosine; similarity above this baseline is rescaled to [0, 1]
# so one threshold works in both hybrid and lexical-fallback modes. Chosen on the development split.
SEMANTIC_BASELINE = 0.30
TAG_BONUS_WEIGHT = 0.12
ERROR_CODE_BONUS_WEIGHT = 0.30
SEMANTIC_SCORE_WEIGHT = 0.45
TEXT_SCORE_WEIGHT = 1 - SEMANTIC_SCORE_WEIGHT
STOP_WORDS = sorted((ENGLISH_STOP_WORDS - {"not", "no", "never", "without", "twice"}) | {
    "redacted_email", "redacted_secret", "redacted_phone",
})


@dataclass(frozen=True)
class Match:
    ticket: Ticket
    score: float
    matched_terms: tuple[str, ...]
    matched_error_codes: tuple[str, ...] = ()
    text_score: float = 0.0
    matched_tags: tuple[str, ...] = ()
    semantic_score: float | None = None


@dataclass(frozen=True)
class SearchAssessment:
    matches: tuple[Match, ...]
    closest_score: float | None
    candidate_count: int


def similarity_level(score: float) -> str:
    if score >= 0.8:
        return "High similarity"
    if score >= 0.5:
        return "Moderate similarity"
    return "Limited similarity"


def search_text(subject: str, description: str) -> str:
    return f"{subject} {subject} {description}"


def sparse_scores(matrix, vector) -> np.ndarray:
    # TfidfVectorizer rows are L2-normalized, so a sparse dot product equals cosine similarity
    # without re-normalizing a copy of the whole corpus matrix on every query.
    return np.asarray((matrix @ vector.T).todense(), dtype=np.float64).ravel()


def top_contributions(row, vector, names: np.ndarray, limit: int = 6) -> list[str]:
    contributions = row.multiply(vector).tocoo()
    strongest = sorted(zip(contributions.col, contributions.data), key=lambda pair: (-pair[1], pair[0]))
    return [str(names[column]) for column, value in strongest[:limit] if value > 0]


@dataclass(frozen=True)
class CandidateScores:
    scores: np.ndarray  # aligned with the candidate indices passed to SimilarityIndex.score
    text_scores: np.ndarray  # aligned with SimilarityIndex.indexed
    semantic_scores: np.ndarray | None
    tag_scores: np.ndarray
    query_vector: object
    tag_query: object
    exact_codes: list[tuple[str, ...]]


# Why a precedent an agent expected was not shown; stored with missed-precedent reports.
ABSENCE_REASONS = ("not_indexed", "outside_review_window", "category_filter", "error_code_conflict",
                   "below_threshold", "ranked_below_shown", "shown")


class SimilarityIndex:
    def __init__(self, tickets: Iterable[Ticket], max_age_days: int = 0, today: date | None = None,
                 semantic_encoder=None, embedding_cache: dict[str, np.ndarray] | None = None) -> None:
        self.max_age_days = validate_age(max_age_days)
        self.today = today
        self.semantic_weight = SEMANTIC_SCORE_WEIGHT
        # Every reusable resolution is indexed once; review-age eligibility is applied per query,
        # so the passing of time never forces a rebuild.
        self.indexed = tuple(sorted((ticket for ticket in tickets if self.reusable(ticket)), key=lambda ticket: ticket.ticket_id))
        self.positions = {ticket.ticket_id: index for index, ticket in enumerate(self.indexed)}
        self.reviewed_days = np.array([date.fromisoformat(ticket.reviewed_at).toordinal() if ticket.reviewed_at else -1
                                       for ticket in self.indexed], dtype=np.int64)
        self.vectorizer = TfidfVectorizer(
            stop_words=STOP_WORDS,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.matrix = None
        self.feature_names = np.array([])
        self.tag_vectorizer = None
        self.tag_matrix = None
        self.tag_names = np.array([])
        self.resolution_matrix = None
        self.semantic_encoder = None
        self.semantic_matrix = None
        self.semantic_warning = None
        self.code_groups = tuple(group_codes(ticket.error_codes) for ticket in self.indexed)
        if self.indexed:
            texts = [search_text(ticket.subject, ticket.description) for ticket in self.indexed]
            self.matrix, self.feature_names = fit_tfidf(self.vectorizer, texts)
            tag_documents = [" ".join(ticket.tags) for ticket in self.indexed]
            if any(tag_documents):
                self.tag_vectorizer = TfidfVectorizer(
                    stop_words=STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode",
                )
                self.tag_matrix, self.tag_names = fit_tfidf(self.tag_vectorizer, tag_documents)
            self.resolution_matrix, _ = fit_tfidf(TfidfVectorizer(
                stop_words=STOP_WORDS, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode",
            ), [ticket.resolution_notes for ticket in self.indexed])
            try:
                source = semantic_model_source()
                if semantic_encoder is None:
                    if source is None:
                        raise SemanticModelUnavailable(
                            "Semantic model unavailable; set SENTENCE_TRANSFORMER_MODEL to a local model directory. Falling back to TF-IDF and tag matching."
                        )
                    semantic_encoder = load_sentence_model(source)
                self.semantic_encoder = semantic_encoder
                # Only encode texts not already embedded; keys are text hashes so a persisted cache holds no ticket text.
                cache = {} if embedding_cache is None else embedding_cache
                keys = [text_key(text) for text in texts]
                missing = {key: text for key, text in zip(keys, texts) if key not in cache}
                if missing:
                    cache.update(zip(missing, encode_sentences(self.semantic_encoder, list(missing.values()))))
                self.semantic_matrix = np.vstack([cache[key] for key in keys])
            except SemanticModelUnavailable as error:
                self.semantic_warning = str(error)

    @staticmethod
    def reusable(ticket: Ticket) -> bool:
        return ticket.status == "resolved" and bool(ticket.resolution_notes) and ticket.resolution_state == "approved"

    @staticmethod
    def eligible(ticket: Ticket, max_age_days: int, today: date) -> bool:
        if not SimilarityIndex.reusable(ticket):
            return False
        if max_age_days:
            if ticket.reviewed_at is None:
                return False
            age = (today - date.fromisoformat(ticket.reviewed_at)).days
            if not 0 <= age <= max_age_days:
                return False
        return True

    def eligible_mask(self, max_age_days: int | None = None, today: date | None = None) -> np.ndarray:
        max_age = self.max_age_days if max_age_days is None else validate_age(max_age_days)
        if not max_age:
            return np.ones(len(self.indexed), dtype=bool)
        age = (today or self.today or datetime.now(timezone.utc).date()).toordinal() - self.reviewed_days
        return (self.reviewed_days >= 0) & (age >= 0) & (age <= max_age)

    @property
    def tickets(self) -> tuple[Ticket, ...]:
        """Tickets eligible under this index's default review policy."""
        return tuple(ticket for ticket, eligible in zip(self.indexed, self.eligible_mask()) if eligible)

    def candidate_indices(self, query: TicketQuery, categories: frozenset[str] | None = None,
                          exclude_ticket_id: str | None = None, max_age_days: int | None = None,
                          today: date | None = None) -> list[int]:
        eligible = self.eligible_mask(max_age_days, today)
        query_groups = group_codes(query.error_codes)
        return [index for index, ticket in enumerate(self.indexed)
                if eligible[index]
                and ticket.ticket_id != exclude_ticket_id
                and (categories is None or ticket.category in categories)
                and not codes_conflict(query_groups, self.code_groups[index])]

    def score(self, query: TicketQuery, candidate_indices: list[int]) -> CandidateScores:
        candidates = np.asarray(candidate_indices, dtype=np.intp)
        query_codes = set(query.error_codes)
        query_text = search_text(query.subject, query.description)
        query_vector = self.vectorizer.transform([query_text]) if self.matrix is not None else None
        text_scores = sparse_scores(self.matrix, query_vector) if query_vector is not None else np.zeros(len(self.indexed))
        semantic_scores = None
        if self.semantic_encoder is not None and self.semantic_matrix is not None:
            try:
                query_embedding = encode_sentences(self.semantic_encoder, [query_text])[0]
                cosine = (self.semantic_matrix @ query_embedding).astype(np.float64)
                semantic_scores = np.clip((cosine - SEMANTIC_BASELINE) / (1 - SEMANTIC_BASELINE), 0, 1)
            except SemanticModelUnavailable as error:
                self.semantic_warning = str(error)
                self.semantic_matrix = None
                self.semantic_encoder = None
        tag_query = None
        tag_scores = np.zeros(len(self.indexed))
        if self.tag_matrix is not None:
            tag_query = self.tag_vectorizer.transform([query_text])
            tag_scores = sparse_scores(self.tag_matrix, tag_query)
        if semantic_scores is not None:
            scores = (1 - self.semantic_weight) * text_scores[candidates] + self.semantic_weight * semantic_scores[candidates]
        else:
            scores = text_scores[candidates].copy()
        scores += TAG_BONUS_WEIGHT * tag_scores[candidates] * (1 - scores)
        exact_codes = [tuple(sorted(query_codes.intersection(self.indexed[index].error_codes))) if query_codes else ()
                       for index in candidate_indices]
        has_code = np.fromiter((bool(codes) for codes in exact_codes), dtype=bool, count=len(exact_codes))
        scores[has_code] += ERROR_CODE_BONUS_WEIGHT * (1 - scores[has_code])
        return CandidateScores(scores, text_scores, semantic_scores, tag_scores, query_vector, tag_query, exact_codes)

    def explain_absence(self, query: TicketQuery, ticket_id: str, min_score: float = DEFAULT_MIN_SCORE, top_n: int = 3,
                        categories: frozenset[str] | None = None, exclude_ticket_id: str | None = None,
                        max_age_days: int | None = None, today: date | None = None) -> tuple[str, float | None]:
        """Why a precedent was not among the shown results, and its score when it was a candidate."""
        position = self.positions.get(ticket_id)
        if position is None:
            return "not_indexed", None
        if not self.eligible_mask(max_age_days, today)[position]:
            return "outside_review_window", None
        if categories is not None and self.indexed[position].category not in categories:
            return "category_filter", None
        if codes_conflict(group_codes(query.error_codes), self.code_groups[position]):
            return "error_code_conflict", None
        candidates = self.candidate_indices(query, categories, exclude_ticket_id, max_age_days, today)
        scores = self.score(query, candidates).scores
        score = float(scores[candidates.index(position)])
        if score <= 0 or score < min_score:
            return "below_threshold", score
        # Same ordering as assess: higher score first, ticket ID breaks ties.
        rank = 1 + sum(1 for other, value in zip(candidates, scores) if value > score or (value == score and other < position))
        return ("ranked_below_shown" if rank > top_n else "shown"), score

    def search(
        self, query: TicketQuery, top_n: int = 3, min_score: float = DEFAULT_MIN_SCORE,
        categories: frozenset[str] | None = None,
        exclude_ticket_id: str | None = None,
        max_age_days: int | None = None, today: date | None = None,
    ) -> list[Match]:
        return list(self.assess(query, top_n, min_score, categories, exclude_ticket_id, max_age_days, today).matches)

    def assess(
        self, query: TicketQuery, top_n: int = 3, min_score: float = DEFAULT_MIN_SCORE,
        categories: frozenset[str] | None = None,
        exclude_ticket_id: str | None = None,
        max_age_days: int | None = None, today: date | None = None,
    ) -> SearchAssessment:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 10:
            raise ValueError("Result limit must be an integer between 1 and 10.")
        if isinstance(min_score, bool) or not isinstance(min_score, (float, int)) or not np.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("Minimum similarity must be between 0 and 1.")
        if categories is not None and (not isinstance(categories, frozenset) or any(not isinstance(category, str) for category in categories)):
            raise ValueError("Categories must be a frozenset of category names or None.")
        if exclude_ticket_id is not None and not isinstance(exclude_ticket_id, str):
            raise ValueError("Excluded ticket ID must be a string or None.")
        candidate_indices = self.candidate_indices(query, categories, exclude_ticket_id, max_age_days, today)
        if (self.matrix is None and self.semantic_matrix is None) or not candidate_indices:
            return SearchAssessment((), None, len(candidate_indices))
        scored = self.score(query, candidate_indices)
        scores, text_scores, semantic_scores = scored.scores, scored.text_scores, scored.semantic_scores
        query_vector, tag_query, tag_scores, exact_codes = scored.query_vector, scored.tag_query, scored.tag_scores, scored.exact_codes
        candidates = np.asarray(candidate_indices, dtype=np.intp)
        # Candidates are in ticket-ID order, so a stable sort keeps ticket ID as the tie-breaker.
        order = np.argsort(-scores, kind="stable")
        matches = []
        for position in order[:top_n]:
            score = float(scores[position])
            if score <= 0 or score < min_score:
                break
            index = int(candidates[position])
            terms = top_contributions(self.matrix[index], query_vector, self.feature_names) if query_vector is not None else []
            matched_tags = []
            if tag_query is not None and tag_scores[index] > 0:
                matched_tags = top_contributions(self.tag_matrix[index], tag_query, self.tag_names)
            matches.append(Match(self.indexed[index], score, tuple(terms), exact_codes[position],
                                 float(text_scores[index]), tuple(matched_tags),
                                 float(semantic_scores[index]) if semantic_scores is not None else None))
        return SearchAssessment(tuple(matches), float(scores[order[0]]), len(candidate_indices))


def validate_age(max_age_days: int) -> int:
    if isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or max_age_days < 0:
        raise ValueError("Resolution age limit must be a nonnegative integer.")
    return max_age_days


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fit_tfidf(vectorizer: TfidfVectorizer, documents: list[str]):
    try:
        return vectorizer.fit_transform(documents), vectorizer.get_feature_names_out()
    except ValueError as error:
        if "empty vocabulary" not in str(error):
            raise
        return None, np.array([])


def group_codes(codes: Iterable[str]) -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = {}
    for code in codes:
        groups.setdefault(code_family(code), set()).add(code)
    return {family: frozenset(members) for family, members in groups.items()}


def codes_conflict(query_groups: dict[str, frozenset[str]], ticket_groups: dict[str, frozenset[str]]) -> bool:
    # A conflict exists when both sides record codes in the same family but share none of them,
    # e.g. ORA-00942 vs ORA-01017, even if another family (HTTP 403) matches exactly.
    return any(family in ticket_groups and not codes.intersection(ticket_groups[family])
               for family, codes in query_groups.items())


def code_family(code: str) -> str:
    if code.startswith(("HTTP ", "SQLSTATE ")):
        return code.split(" ", 1)[0]
    return code.split("-", 1)[0]