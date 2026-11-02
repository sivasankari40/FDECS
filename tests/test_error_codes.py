import pytest

from support_assistant.config import ROOT
from support_assistant.error_codes import extract_error_codes, normalize_error_code, parse_error_code_input
from support_assistant.matching import SimilarityIndex
from support_assistant.models import Ticket, TicketQuery
from support_assistant.repository import load_history


DATA = ROOT / "starter_kit" / "data" / "tickets.json"


def resolved_ticket(ticket_id, subject, description, tags=(), error_codes=()):
    return Ticket.from_dict({
        "ticket_id": ticket_id,
        "category": "api",
        "subject": subject,
        "description": description,
        "resolution_notes": "Reviewed synthetic resolution.",
        "tags": list(tags),
        "created_at": "2026-10-01",
        "status": "resolved",
        "error_codes": list(error_codes),
    })


@pytest.mark.parametrize("value,expected", [
    ("415", "HTTP 415"), ("http-403", "HTTP 403"), ("SQLSTATE 42501", "SQLSTATE 42501"),
    ("ora-00942", "ORA-00942"), ("ERR_CONN_RESET", "ERR-CONN_RESET"), ("0x00AF11", "0X00AF11"),
])
def test_error_code_normalization(value, expected):
    assert normalize_error_code(value) == expected


@pytest.mark.parametrize("value", ["", "timeout", "2026", "HTTP 999", "not an error code"])
def test_invalid_error_code_rejected(value):
    with pytest.raises(ValueError):
        normalize_error_code(value)


def test_parse_multiple_codes_deduplicates_and_normalizes():
    assert parse_error_code_input("http 403, 403; ORA-00942") == ("HTTP 403", "ORA-00942")


def test_error_codes_extracted_from_explicit_historical_mentions_and_tags():
    history = load_history(DATA) + load_history(ROOT / "data" / "production_tickets.json")
    codes = {ticket.ticket_id: ticket.error_codes for ticket in history if ticket.error_codes}
    assert codes["T-001702"] == ("HTTP 429",)
    assert codes["T-001703"] == ("HTTP 500",)
    assert codes["T-002004"] == ("HTTP 415",)
    assert codes["T-002011"] == ("HTTP 403",)
    assert extract_error_codes(["Customer has 5 active seats in 2026"]) == ()


def test_tags_add_low_weight_context_without_changing_primary_text():
    shared_subject = "Connector integration fails"
    shared_description = "The source connection returns an unclear error."
    tagged = resolved_ticket("T-009001", shared_subject, shared_description, ("webhook", "signature", "hmac"))
    unrelated = resolved_ticket("T-009002", shared_subject, shared_description, ("invoice", "discount", "checkout"))
    index = SimilarityIndex([unrelated, tagged])
    assessment = index.assess(TicketQuery(shared_subject, shared_description + " webhook signature verification"), min_score=0)
    assert [match.ticket.ticket_id for match in assessment.matches] == ["T-009001", "T-009002"]
    assert "webhook" in assessment.matches[0].matched_tags
    assert assessment.matches[0].text_score == assessment.matches[1].text_score


def test_exact_error_code_ranks_same_symptom_ahead_of_other_code():
    common_subject = "API request rejected"
    common_description = "The client receives an error when calling the endpoint."
    timeout = resolved_ticket("T-009003", common_subject, common_description, ("api", "request"), ("HTTP 504",))
    forbidden = resolved_ticket("T-009004", common_subject, common_description + " Forbidden authorization response.", ("api", "request"), ("HTTP 403",))
    index = SimilarityIndex([timeout, forbidden])
    assessment = index.assess(TicketQuery(common_subject, common_description, ("403",)))
    assert assessment.matches[0].ticket.ticket_id == "T-009004"
    assert assessment.matches[0].matched_error_codes == ("HTTP 403",)
    assert "HTTP 403" in assessment.matches[0].matched_error_codes
    assert len(assessment.matches) == 1


def test_unmatched_error_code_does_not_count_as_exact_evidence():
    ticket = resolved_ticket("T-009005", "API request rejected", "Client gets an endpoint error.", error_codes=("HTTP 403",))
    assessment = SimilarityIndex([ticket]).assess(TicketQuery("API request rejected", "Client gets an endpoint error.", ("HTTP 500",)))
    assert assessment.matches == ()
    assert assessment.candidate_count == 0


def test_error_code_filter_does_not_exclude_candidates_without_code_metadata():
    ticket = resolved_ticket("T-009006", "API request rejected", "Client gets an endpoint error.")
    assessment = SimilarityIndex([ticket]).assess(TicketQuery("API request rejected", "Client gets an endpoint error.", ("HTTP 500",)))
    assert assessment.candidate_count == 1


def test_error_code_metadata_backfill_is_deterministic():
    old_document = {"ticket_id": "T-002004", "category": "api", "subject": "API rejects JSON",
                    "description": "The request gets HTTP 415 unsupported media type.",
                    "resolution_notes": "Set the JSON content type.", "tags": [], "created_at": "2026-09-06",
                    "status": "resolved"}
    ticket = Ticket.from_dict(old_document)
    updated = {**old_document, **ticket.to_dict()}
    assert ticket.error_codes == ("HTTP 415",)
    assert Ticket.from_dict(updated).error_codes == ("HTTP 415",)


def test_unqualified_numbers_are_not_error_codes():
    old_document = {"ticket_id": "T-002003", "category": "api", "subject": "Account report has 500 rows",
                    "description": "The request in 2026 took 3 minutes and returned 403 characters.",
                    "resolution_notes": "Review the report.", "tags": [], "created_at": "2026-09-06",
                    "status": "resolved"}
    assert Ticket.from_dict(old_document).error_codes == ()


MULTI_CODE_TICKETS = {
    "T-009101": ("HTTP 403",),
    "T-009102": ("HTTP 500",),
    "T-009103": ("HTTP 403", "HTTP 500"),
    "T-009104": ("ORA-00942",),
    "T-009105": (),
    "T-009106": ("HTTP 403", "ORA-01017"),
    "T-009107": ("HTTP 502",),
    "T-009108": ("SQLSTATE 42501", "HTTP 403"),
}


@pytest.fixture
def multi_code_index():
    return SimilarityIndex([resolved_ticket(ticket_id, "Report API request rejected", "Client receives an error from the report endpoint.",
                                            error_codes=codes) for ticket_id, codes in MULTI_CODE_TICKETS.items()])


# Expected candidates per query: a ticket is excluded only when it records a code in a family
# the query also uses and shares no code with the query in that family.
@pytest.mark.parametrize("query_codes,excluded,boosted", [
    (("HTTP 403",), {"T-009102", "T-009107"}, {"T-009101", "T-009103", "T-009106", "T-009108"}),
    (("HTTP 403", "HTTP 500"), {"T-009107"}, {"T-009101", "T-009102", "T-009103", "T-009106", "T-009108"}),
    (("HTTP 403", "ORA-00942"), {"T-009102", "T-009106", "T-009107"}, {"T-009101", "T-009103", "T-009104", "T-009108"}),
    (("HTTP 503",), {"T-009101", "T-009102", "T-009103", "T-009106", "T-009107", "T-009108"}, set()),
    (("SQLSTATE 42P01",), {"T-009108"}, set()),
    ((), set(), set()),
])
def test_multi_code_tickets_are_filtered_per_code_family(multi_code_index, query_codes, excluded, boosted):
    query = TicketQuery("Report API request fails", "Client gets an error from the endpoint.", query_codes)
    assessment = multi_code_index.assess(query, top_n=10, min_score=0)
    returned = {match.ticket.ticket_id: match for match in assessment.matches}
    assert set(returned) == set(MULTI_CODE_TICKETS) - excluded
    assert {ticket_id for ticket_id, match in returned.items() if match.matched_error_codes} == boosted
    unboosted = [match.score for ticket_id, match in returned.items() if ticket_id not in boosted]
    assert all(returned[ticket_id].score > max(unboosted, default=0) for ticket_id in boosted)


def test_matching_code_in_one_family_does_not_hide_conflict_in_another(multi_code_index):
    query = TicketQuery("Report API request fails", "Client gets an error from the endpoint.", ("HTTP 403", "ORA-00942"))
    returned = {match.ticket.ticket_id for match in multi_code_index.search(query, top_n=10, min_score=0)}
    assert "T-009106" not in returned  # HTTP 403 matches, but ORA-01017 contradicts ORA-00942
    assert "T-009103" in returned  # a recorded HTTP 500 alongside the matching HTTP 403 is not a conflict
