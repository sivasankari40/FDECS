from time import perf_counter

import streamlit as st

from support_assistant.config import Settings
from support_assistant.consistency import conflicts_among
from support_assistant.error_codes import parse_error_code_input
from support_assistant.feedback import FeedbackStore
from support_assistant.interactions import CATEGORY_FILTERS, InteractionHistory, category_codes
from support_assistant.matching import Match, similarity_level
from support_assistant.models import TicketQuery
from support_assistant.ticket_ids import canonical_ticket_id
from support_assistant.observability import event
from support_assistant.privacy import display_text
from support_assistant.presentation import (absence_explanation, load_triage_routing, match_reasons, no_match_advice,
                                             resolution_draft, term_badges)
from support_assistant.repository import CouchRepository, RepositoryError
from support_assistant.security import SessionRateLimiter, identity_allowed, validate_oidc_config
from support_assistant.service import EMBEDDING_CACHE_DIRECTORY, IndexStore, Snapshot


st.set_page_config(page_title="Solari | Support", page_icon=":material/support_agent:", layout="wide")
st.markdown("""
<style>
    .stMainBlockContainer {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
    h1 {font-size: 2rem !important; letter-spacing: 0 !important;}
    h2 {font-size: 1.35rem !important; letter-spacing: 0 !important;}
    h3 {font-size: 1.1rem !important; letter-spacing: 0 !important; overflow-wrap: anywhere;}
    [data-testid="stVerticalBlockBorderWrapper"] {border-radius: 6px;}
    [data-testid="stMetricValue"] {font-size: 1.25rem;}
    .term-list {display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 12px;}
    .term-chip {display: inline-block; max-width: 100%; overflow-wrap: anywhere;
        border: 1px solid #b7dcc8; border-radius: 4px; background: #e8f6ee;
        color: #135c45; padding: 3px 8px; font-size: 0.8rem; letter-spacing: 0;}
    @media (max-width: 640px) {
        .stMainBlockContainer {padding: 1rem;}
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False, max_entries=4)
def shared_store(namespace: tuple[str, str, str]) -> IndexStore:
    return IndexStore(cache_directory=EMBEDDING_CACHE_DIRECTORY, background=True)


def require_access(settings: Settings) -> None:
    if settings.environment != "production":
        return
    try:
        auth = dict(st.secrets["auth"])
        validate_oidc_config(auth)
    except (KeyError, FileNotFoundError, ValueError):
        st.error("Production authentication is not configured correctly.")
        st.stop()
    claims = dict(st.user)
    if claims.get("is_logged_in") is not True:
        st.info("Sign in to continue.")
        if st.button("Sign in", icon=":material/login:"):
            st.login()
        st.stop()
    if not identity_allowed(claims, settings, auth["client_id"]):
        event("access_denied")
        st.error("Access denied or session expired.")
        if st.button("Sign out", icon=":material/logout:"):
            st.logout()
        st.stop()


def clear_results() -> None:
    for key in ("results", "searched", "search_error", "current_search_id", "result_fingerprint", "last_categories",
                "assessment", "resolution_draft", "draft_source", "conflicts", "feedback_counts", "feedback_save_error",
                "missed_message"):
        st.session_state.pop(key, None)


def clear_search_history() -> None:
    st.session_state.interactions.clear()
    st.session_state.pop("replay_search", None)
    clear_results()


def reopen_search(search_id: str) -> None:
    try:
        entry = st.session_state.interactions.find(search_id)
        st.session_state.ticket_id = entry.incoming_ticket_id or ""
        st.session_state.pop("ticket_lookup_pending", None)
        st.session_state.pop("ticket_lookup_error", None)
        st.session_state.subject = entry.query.subject
        st.session_state.description = entry.query.description
        st.session_state.error_code_input = ", ".join(entry.query.error_codes)
        st.session_state.categories = list(entry.categories)
        loaded = st.session_state.get("loaded_ticket")
        if loaded is None or loaded.ticket_id != st.session_state.ticket_id:
            st.session_state.pop("loaded_ticket", None)
        clear_results()
        st.session_state.replay_search = True
    except ValueError as error:
        st.session_state.search_error = str(error)


def ticket_id_changed() -> None:
    clear_results()
    st.session_state.pop("ticket_lookup_error", None)
    st.session_state.pop("loaded_ticket", None)
    st.session_state.ticket_lookup_pending = bool(st.session_state.ticket_id.strip())
    if st.session_state.ticket_lookup_pending:
        st.session_state.subject = ""
        st.session_state.description = ""
        st.session_state.error_code_input = ""


def clear_ticket() -> None:
    st.session_state.ticket_id = ""
    st.session_state.subject = ""
    st.session_state.description = ""
    st.session_state.error_code_input = ""
    ticket_id_changed()


def request_search() -> None:
    st.session_state.search_pending = True


def reuse_resolution(match: Match) -> None:
    st.session_state.resolution_draft = resolution_draft(match)
    st.session_state.draft_source = match.ticket.ticket_id


def load_current_snapshot(settings: Settings, force: bool = False) -> Snapshot:
    reader = settings.for_reader()
    namespace = (reader.url, reader.database, reader.username)
    with CouchRepository(reader) as repository:
        snapshot = shared_store(namespace).refresh(repository, force=force)
    if any(ticket.status != "resolved" for ticket in snapshot.tickets):
        raise RepositoryError("Historical database contains nonresolved tickets. Move them to the incoming database before searching.")
    st.session_state.snapshot = snapshot
    st.session_state.namespace = namespace
    return snapshot


def lookup_incoming_ticket(settings: Settings, ticket_id: str):
    with CouchRepository(settings.for_incoming()) as repository:
        return repository.get_ticket(ticket_id)


def load_feedback_counts(settings: Settings, ticket_ids: list[str]) -> dict[str, tuple[int, int]]:
    try:
        with FeedbackStore(settings) as store:
            return store.counts(ticket_ids)
    except (RepositoryError, ValueError):
        return {}


def report_missed_precedent(settings: Settings, search_id: str) -> None:
    entry = st.session_state.interactions.find(search_id)
    field = "missed_" + search_id
    try:
        ticket_id = canonical_ticket_id(st.session_state.get(field, ""))
        if ticket_id in entry.result_ids:
            raise ValueError(f"{ticket_id} is already shown above; use its helpful/not helpful buttons instead.")
        if ticket_id not in {ticket.ticket_id for ticket in st.session_state.snapshot.tickets}:
            raise ValueError(f"{ticket_id} is not in the resolved ticket history.")
        index = shared_store(st.session_state.namespace).index
        reason, score = index.explain_absence(entry.query, ticket_id, settings.min_score, categories=category_codes(list(entry.categories)),
                                              exclude_ticket_id=entry.incoming_ticket_id, max_age_days=settings.resolution_age_days)
        if reason == "shown":
            raise ValueError(f"{ticket_id} would now be shown; the history changed since this search. Search again.")
        with FeedbackStore(settings) as store:
            store.record_missed(entry.search_id, ticket_id, entry.incoming_ticket_id, entry.result_ids, reason, score)
        st.session_state.missed_message = ("success", f"Recorded {ticket_id} as a missed precedent. It was not shown because "
                                           + absence_explanation(reason, score, settings.min_score))
        st.session_state[field] = ""
    except ValueError as error:
        st.session_state.missed_message = ("error", str(error))
    except RepositoryError:
        st.session_state.missed_message = ("warning", "The feedback database is unavailable; the missed precedent was not recorded.")


def save_feedback(settings: Settings, entry, rank: int, match: Match, vote: int | None) -> None:
    try:
        with FeedbackStore(settings) as store:
            store.record(entry.search_id, match.ticket.ticket_id, entry.incoming_ticket_id,
                         None if vote is None else vote == 1, rank, match.score)
        st.session_state.feedback_save_error = None
    except (RepositoryError, ValueError):
        st.session_state.feedback_save_error = "The feedback database is unavailable; this vote is kept for this session only."


st.session_state.setdefault("ticket_id", "")
st.session_state.setdefault("subject", "")
st.session_state.setdefault("description", "")
st.session_state.setdefault("error_code_input", "")
st.session_state.setdefault("categories", [])
st.session_state.setdefault("interactions", InteractionHistory())
st.session_state.setdefault("limiter", SessionRateLimiter())
st.title("Solari Support")
st.caption("TICKET SIMILARITY WORKSPACE")
st.divider()
settings = None
database_error = None
LOOKUP_UNAVAILABLE = "Ticket lookup is unavailable while the incoming-ticket database is offline."
try:
    settings = Settings.from_env(role="reader")
    require_access(settings)
    snapshot = load_current_snapshot(settings)
    store = shared_store(st.session_state.namespace)
    index = store.index_for(snapshot)
    routing = load_triage_routing(settings.triage_routing_file)
    if st.session_state.get("interaction_namespace") != st.session_state.namespace:
        clear_search_history()
        st.session_state.interaction_namespace = st.session_state.namespace
except (ValueError, RepositoryError) as error:
    database_error = str(error)
    st.session_state.pop("results", None)
    st.session_state.pop("searched", None)

if st.session_state.pop("ticket_lookup_pending", False):
    ticket_id = st.session_state.ticket_id.strip()
    try:
        if settings is None:
            raise RepositoryError("Configuration unavailable.")
        entered_id = ticket_id.upper()
        ticket_id = canonical_ticket_id(ticket_id)
        ticket = lookup_incoming_ticket(settings, ticket_id)
        if ticket is None:
            st.session_state.ticket_lookup_error = f"Ticket ID {ticket_id} was not found in the incoming-ticket database."
        elif entered_id != ticket_id and entered_id not in {alias.strip().upper() for alias in ticket.legacy_ids}:
            st.session_state.ticket_lookup_error = "That legacy alias is not registered for this ticket. Use its canonical Ticket ID."
        else:
            st.session_state.ticket_id = ticket_id
            st.session_state.subject = ticket.subject
            st.session_state.description = ticket.description
            st.session_state.error_code_input = ", ".join(ticket.error_codes)
            st.session_state.loaded_ticket = ticket
    except ValueError as error:
        st.session_state.ticket_lookup_error = str(error)
    except RepositoryError:
        st.session_state.ticket_lookup_error = LOOKUP_UNAVAILABLE

search_requested = st.session_state.pop("search_pending", False)
loaded_ticket = st.session_state.get("loaded_ticket")

with st.sidebar:
    st.subheader("Ticket workspace")
    if settings is None or settings.environment == "development":
        st.caption("LOCAL DEVELOPMENT")
    elif st.button("Sign out", icon=":material/logout:", width="stretch"):
        st.logout()
    selected_categories = st.pills("Category", list(CATEGORY_FILTERS), selection_mode="multi", key="categories", disabled=bool(database_error)) or []
    st.divider()
    st.caption("KNOWLEDGE BASE")
    if database_error:
        st.write(":material/cloud_off: Unavailable")
    else:
        st.metric("Eligible resolutions", int(index.eligible_mask(settings.resolution_age_days).sum()))
        st.caption(f"{len(snapshot.tickets)} resolved tickets")
        st.caption(settings.database)
        if store.rebuilding:
            st.caption(":material/sync: Updating the search index with recent changes; searches use the previous index until it finishes.")
    if st.button("Refresh history", icon=":material/refresh:", width="stretch", disabled=settings is None):
        try:
            snapshot = load_current_snapshot(settings, force=True)
            clear_results()
            st.rerun()
        except RepositoryError as error:
            st.error(str(error))

if database_error:
    st.error(database_error, icon=":material/cloud_off:")
elif not snapshot.tickets:
    st.warning("Ticket history is empty.", icon=":material/inbox:")
if st.session_state.get("ticket_lookup_error") == LOOKUP_UNAVAILABLE:
    st.warning("Incoming-ticket lookup is unavailable. Manual subject/description search is still available.")
if not database_error and index.semantic_warning:
    st.warning(index.semantic_warning, icon=":material/psychology_alt:")

st.subheader("Incoming ticket")
with st.container(border=True):
    st.button("Clear ticket", icon=":material/clear_all:", on_click=clear_ticket)
    st.text_input("Ticket ID", key="ticket_id", max_chars=100, on_change=ticket_id_changed,
                  placeholder="Required", help="Required for every search. An incoming ticket ID can auto-fill its details.")
    if loaded_ticket:
        st.caption(f"{loaded_ticket.ticket_id} | {loaded_ticket.status.replace('_', ' ').title()} | Incoming")
    if st.session_state.get("ticket_lookup_error"):
        st.warning(st.session_state.ticket_lookup_error)
    st.text_input("Subject", key="subject", max_chars=300)
    st.text_area("Description", key="description", max_chars=10000, height=150)
    st.text_input("Error code (optional)", key="error_code_input", max_chars=200,
                  help="Enter exact codes separated by commas, e.g. HTTP 403, ORA-00942, or SQLSTATE 42501.")
    ticket_lookup_error = st.session_state.get("ticket_lookup_error", "")
    lookup_blocks_search = bool(ticket_lookup_error) and ticket_lookup_error != LOOKUP_UNAVAILABLE
    submitted = st.button("Find similar tickets", icon=":material/search:", type="primary", on_click=request_search,
                          disabled=bool(database_error) or lookup_blocks_search)

replay = st.session_state.pop("replay_search", False)
filters_changed = "searched" in st.session_state and tuple(selected_categories) != st.session_state.get("last_categories", ())
if submitted or search_requested or replay or filters_changed:
    previous_query = st.session_state.get("searched")
    clear_results()
    try:
        if not st.session_state.ticket_id.strip():
            raise ValueError("Ticket ID cannot be empty.")
        query = TicketQuery(*previous_query) if filters_changed and not submitted and not replay else TicketQuery(
            st.session_state.subject, st.session_state.description, parse_error_code_input(st.session_state.error_code_input))
        if database_error:
            raise ValueError(database_error)
        if not st.session_state.limiter.allow(settings.search_limit_per_minute):
            raise ValueError("Search limit reached. Try again in one minute.")
        with st.spinner("Searching ticket history..."):
            # The snapshot and index were already refreshed from CouchDB at the top of this rerun.
            start = perf_counter()
            assessment = index.assess(query, min_score=settings.min_score, categories=category_codes(selected_categories),
                                      exclude_ticket_id=st.session_state.ticket_id.strip() or None,
                                      max_age_days=settings.resolution_age_days)
            st.session_state.assessment = assessment
            st.session_state.results = list(assessment.matches)
            st.session_state.search_ms = round((perf_counter() - start) * 1000, 2)
            event("ticket_search", result_count=len(st.session_state.results), elapsed_ms=st.session_state.search_ms)
            st.session_state.searched = (query.subject, query.description, query.error_codes)
            st.session_state.last_categories = tuple(selected_categories)
            st.session_state.result_fingerprint = index.fingerprint
            st.session_state.conflicts = conflicts_among(index, [match.ticket for match in st.session_state.results])
            st.session_state.feedback_counts = load_feedback_counts(settings, [match.ticket.ticket_id for match in st.session_state.results])
            entry = st.session_state.interactions.record(query, selected_categories,
                tuple(match.ticket.ticket_id for match in st.session_state.results), index.fingerprint,
                incoming_ticket_id=st.session_state.ticket_id.strip() or None)
            st.session_state.current_search_id = entry.search_id
    except (ValueError, RepositoryError) as error:
        st.session_state.search_error = str(error)

if not database_error and "results" in st.session_state and st.session_state.get("result_fingerprint") != index.fingerprint:
    clear_results()

with st.sidebar:
    st.divider()
    st.subheader("Search history")
    if not st.session_state.interactions.entries:
        st.caption("No searches yet.")
    else:
        with st.container(height=280, border=False):
            for entry in st.session_state.interactions.entries:
                label = entry.query.subject[:52] + ("..." if len(entry.query.subject) > 52 else "")
                st.button(display_text(label), icon=":material/history:", key="history_" + entry.search_id,
                          on_click=reopen_search, args=(entry.search_id,), width="stretch", disabled=bool(database_error),
                          help=entry.query.subject)
                count_label = "match" if len(entry.result_ids) == 1 else "matches"
                st.caption(display_text(f"{entry.searched_at[11:16]} UTC | {', '.join(entry.categories) or 'All categories'} | {len(entry.result_ids)} {count_label}"))
    st.button("Clear search history", icon=":material/delete_sweep:", on_click=clear_search_history,
              width="stretch", disabled=not st.session_state.interactions.entries)

if st.session_state.get("search_error"):
    st.error(st.session_state.search_error)

if "results" in st.session_state:
    st.divider()
    results = st.session_state.results
    current_entry = st.session_state.interactions.find(st.session_state.current_search_id)
    assessment = st.session_state.assessment
    st.subheader("Top 3 similar tickets")
    count_label = "match" if len(results) == 1 else "matches"
    st.caption(f"{len(results)} qualifying {count_label} from {assessment.candidate_count} eligible candidates.")
    st.caption(display_text(f"Results for: {current_entry.query.subject} | {', '.join(current_entry.categories) or 'All categories'}"))
    if not results:
        st.info("No strong historical match found.", icon=":material/manage_search:")
        if assessment.closest_score is not None:
            st.caption(f"Closest match score: {assessment.closest_score:.1%} similarity. Minimum for recommendation: {settings.min_score:.1%}.")
        else:
            st.caption("No comparable eligible tickets under the current filters and review policy.")
        advice = no_match_advice(current_entry.query, current_entry.categories, routing)
        st.markdown("**Recommended action**")
        for action in advice.actions:
            st.markdown("- " + display_text(action))
    else:
        st.caption("Historical resolutions are precedents, not confirmed diagnoses.")
        for conflict in st.session_state.get("conflicts", ()):
            st.warning(display_text(
                f"{conflict.first.ticket_id} and {conflict.second.ticket_id} describe similar problems but were resolved differently. "
                "Confirm which cause applies to this customer before replying."), icon=":material/rule:")
        for rank, match in enumerate(results, start=1):
            with st.container(border=True):
                st.caption(display_text(f"MATCH {rank}  |  {match.ticket.ticket_id}  |  {match.ticket.category.upper()}  |  {match.ticket.created_at}"))
                title_column, score_column = st.columns([5, 1], vertical_alignment="center")
                with title_column:
                    st.subheader(display_text(match.ticket.subject))
                    st.caption(similarity_level(match.score))
                with score_column:
                    st.metric("Evidence score", f"{match.score:.1%}", help=f"Blended text/semantic/tag/code ranking signal; not a confidence probability. Text-only similarity: {match.text_score:.1%}.")
                if match.semantic_score is not None:
                    st.caption(f"Semantic similarity: {match.semantic_score:.1%}")
                st.caption("Shared subject/description terms")
                st.markdown(term_badges(match.matched_terms), unsafe_allow_html=True)
                if match.matched_tags:
                    st.caption("Matching historical tags")
                    st.markdown(term_badges(match.matched_tags), unsafe_allow_html=True)
                if match.matched_error_codes:
                    st.caption("Exact error-code match")
                    st.markdown(term_badges(match.matched_error_codes), unsafe_allow_html=True)
                with st.expander("Why was this suggested?"):
                    for reason in match_reasons(match, current_entry.categories, settings.min_score):
                        st.markdown("- " + display_text(reason))
                st.write(display_text(match.ticket.description))
                if match.ticket.error_codes:
                    st.caption("Recorded error codes: " + ", ".join(match.ticket.error_codes))
                st.markdown("**Historical resolution**")
                st.write(display_text(match.ticket.resolution_notes))
                st.caption(display_text("Tags: " + ", ".join(match.ticket.tags)))
                st.caption(display_text(f"Reviewed: {match.ticket.reviewed_at or 'date unavailable'}  |  Version: {match.ticket.product_version or 'not recorded'}"))
                st.button("Use resolution", icon=":material/post_add:", key=f"reuse_{current_entry.search_id}_{match.ticket.ticket_id}",
                          on_click=reuse_resolution, args=(match,), help="Place this precedent in an editable agent-review draft; nothing is sent.")
                helpful_votes, unhelpful_votes = st.session_state.get("feedback_counts", {}).get(match.ticket.ticket_id, (0, 0))
                if helpful_votes or unhelpful_votes:
                    st.caption(f"Earlier agent feedback: helpful {helpful_votes} | not helpful {unhelpful_votes}")
                st.caption("Was this helpful?")
                previous_vote = current_entry.votes.get(match.ticket.ticket_id)
                vote = st.feedback("thumbs", key=f"feedback_{current_entry.search_id}_{match.ticket.ticket_id}", default=previous_vote)
                st.session_state.interactions.vote(current_entry.search_id, match.ticket.ticket_id, vote)
                if vote != previous_vote:
                    save_feedback(settings, current_entry, rank, match, vote)
                    event("result_feedback", helpful=vote == 1)
                if vote is not None:
                    st.caption("Marked helpful." if vote == 1 else "Marked not helpful.")
    if st.session_state.get("feedback_save_error"):
        st.warning(st.session_state.feedback_save_error)
    with st.expander("Is the right precedent missing?"):
        st.caption("Tell us which resolved ticket should have been suggested. Only its ID is stored.")
        st.text_input("Ticket ID of the right precedent", key="missed_" + current_entry.search_id, max_chars=20, placeholder="e.g. T-001303")
        st.button("Report missed precedent", icon=":material/flag:", key="report_missed_" + current_entry.search_id,
                  on_click=report_missed_precedent, args=(settings, current_entry.search_id))
    if st.session_state.get("missed_message"):
        level, message = st.session_state.missed_message
        getattr(st, level)(display_text(message))
    st.caption(f"Minimum similarity: {settings.min_score:.2f}. Scores are not probabilities. Search service: {st.session_state.search_ms:.2f} ms.")

    if st.session_state.get("draft_source"):
        st.divider()
        st.subheader("Resolution draft")
        st.caption(display_text(f"Source: {st.session_state.draft_source} | Agent review required. Nothing has been sent."))
        st.text_area("Draft text", key="resolution_draft", max_chars=25000, height=180)
        with st.expander("Copy or download draft"):
            st.code(st.session_state.resolution_draft, language=None, wrap_lines=True)
            st.download_button("Download draft", data=st.session_state.resolution_draft,
                               file_name=f"resolution_{st.session_state.draft_source}.txt", mime="text/plain",
                               icon=":material/download:", on_click="ignore")