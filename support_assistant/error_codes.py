import re


_HTTP = re.compile(r"(?i)^(?:HTTP[\s_-]*)?([45][0-9]{2})(?:[\s_-]*ERRORS?)?$")
_HTTP_MENTION = re.compile(r"(?i)\bHTTP(?:\s+STATUS)?\s*[:=]?\s*([45][0-9]{2})\b")
_HTTP_ERROR_MENTION = re.compile(r"(?i)\b([45][0-9]{2})[-_ ]*(?:ERRORS?|RATE[- ]LIMIT|FORBIDDEN|UNSUPPORTED)\b")
_SQLSTATE = re.compile(r"(?i)^SQLSTATE[\s:=_-]*([0-9A-Z]{5})$")
_ORACLE = re.compile(r"(?i)^ORA[-_ ]?(\d{4,6})$")
_HEX = re.compile(r"(?i)^0X([0-9A-F]{4,})$")
_NAMED = re.compile(r"(?i)^(?:ERR(?:OR)?[-_:])([A-Z0-9][A-Z0-9._:-]{1,39})$")


def normalize_error_code(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Error codes must be strings.")
    code = " ".join(value.strip().upper().split())
    for pattern in (_HTTP, _SQLSTATE, _ORACLE, _HEX, _NAMED):
        match = pattern.fullmatch(code)
        if match:
            family = pattern
            if family is _HTTP:
                return "HTTP " + match.group(1)
            if family is _SQLSTATE:
                return "SQLSTATE " + match.group(1)
            if family is _ORACLE:
                return "ORA-" + match.group(1)
            if family is _HEX:
                return "0X" + match.group(1)
            return "ERR-" + match.group(1)
    raise ValueError("Use HTTP 4xx/5xx, SQLSTATE, ORA-xxxxx, ERR-name, or 0x hexadecimal error-code formats.")


def normalize_error_codes(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)) or len(values) > 10:
        raise ValueError("Provide at most 10 error codes.")
    return tuple(sorted({normalize_error_code(value) for value in values}))


def parse_error_code_input(value: str) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise ValueError("Error codes must be text.")
    values = [item.strip() for item in re.split(r"[,;\n]+", value) if item.strip()]
    return normalize_error_codes(values)


def extract_error_codes(texts: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    found = set()
    for text in texts:
        if not isinstance(text, str):
            continue
        for pattern in (_HTTP_MENTION, _HTTP_ERROR_MENTION):
            found.update("HTTP " + match for match in pattern.findall(text))
        for match in re.findall(r"(?i)\bSQLSTATE[\s:=_-]*([0-9A-Z]{5})\b", text):
            found.add("SQLSTATE " + match.upper())
        for match in re.findall(r"(?i)\bORA[-_ ]?(\d{4,6})\b", text):
            found.add("ORA-" + match)
        for match in re.findall(r"(?i)\b(?:ERR|ERROR)[_-]([A-Z0-9][A-Z0-9._:-]{1,39})\b", text):
            found.add("ERR-" + match.upper())
        for match in re.findall(r"(?i)\b0x([0-9a-f]{4,})\b", text):
            found.add("0X" + match.upper())
        for match in re.findall(r"(?i)(?<![A-Z0-9])([45][0-9]{2})-ERROR\b", text):
            found.add("HTTP " + match)
    return tuple(sorted(found))