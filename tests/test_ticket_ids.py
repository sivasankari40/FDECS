import pytest

from support_assistant.ticket_ids import canonical_ticket_id
from support_assistant.models import Ticket


@pytest.mark.parametrize("source,expected", [
    ("T-1303", "T-001303"), ("PT-2001", "T-002001"), ("DEV-01", "T-003001"),
    ("TEST-18", "T-004018"), ("NT-4", "T-005004"), (" t-002001 ", "T-002001"),
])
def test_canonical_and_legacy_ids(source, expected):
    assert canonical_ticket_id(source) == expected
    assert canonical_ticket_id(expected) == expected


@pytest.mark.parametrize("value", ["", "../other", "random", "T-1000000", None])
def test_invalid_ids(value):
    with pytest.raises(ValueError):
        canonical_ticket_id(value)


def test_incoming_ticket_has_no_reusable_resolution():
    ticket = Ticket.incoming("T-005005", "New warehouse issue", "A new connection error")
    assert ticket.status == "open"
    assert ticket.resolution_state == "needs_review"
    assert ticket.resolution_notes == ""
    assert ticket.reviewed_at is None
    assert Ticket.from_dict(ticket.to_dict()) == ticket


def test_legacy_alias_survives_roundtrip():
    ticket = Ticket.incoming("NT-4", "New warehouse issue", "A new connection error")
    assert ticket.ticket_id == "T-005004"
    assert ticket.legacy_ids == ("NT-4",)
    assert Ticket.from_dict(ticket.to_dict()) == ticket


def test_incoming_ticket_extracts_error_codes_from_description():
    ticket = Ticket.incoming("T-005005", "API issue", "The client receives HTTP 403 from the endpoint.")
    assert ticket.error_codes == ("HTTP 403",)