import re


CANONICAL_ID = re.compile(r"T-[0-9]{6}")
LEGACY_ID = re.compile(r"(T|PT|NT|DEV|TEST)-([0-9]+)")
LEGACY_BASES = {"T": 0, "PT": 0, "DEV": 3000, "TEST": 4000, "NT": 5000}


def canonical_ticket_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Ticket ID must be a string.")
    normalized = value.strip().upper()
    if CANONICAL_ID.fullmatch(normalized):
        if normalized == "T-000000":
            raise ValueError("Ticket number must be between 1 and 999999.")
        return normalized
    match = LEGACY_ID.fullmatch(normalized)
    if match is None:
        raise ValueError("Ticket ID must use T-######, for example T-002001.")
    number = int(match.group(2)) + LEGACY_BASES[match.group(1)]
    if not 1 <= number <= 999999:
        raise ValueError("Ticket ID is outside the supported six-digit range.")
    return f"T-{number:06d}"
