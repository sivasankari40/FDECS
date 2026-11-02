import re


EMAIL = re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b")
SECRET = re.compile(r"(?i)\b(api[_-]?key|password|access[_-]?token|refresh[_-]?token|secret|card[_-]?number)\s*[:=]\s*[^\s,;]+")
BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
PHONE = re.compile(r"(?i)\b(phone|mobile)\s*[:=]\s*\+?[\d() .-]{7,25}\d")


def redact_text(text: str) -> str:
    text = EMAIL.sub("[REDACTED_EMAIL]", text)
    text = SECRET.sub(lambda match: match.group(1) + "=[REDACTED_SECRET]", text)
    text = BEARER.sub("Bearer [REDACTED_SECRET]", text)
    return PHONE.sub(lambda match: match.group(1) + "=[REDACTED_PHONE]", text)


def display_text(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+!>|~])", r"\\\1", text)