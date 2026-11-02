import math
import time
from collections import deque
from typing import Any, Mapping
from urllib.parse import urlsplit

from support_assistant.config import Settings


def validate_oidc_config(auth: Mapping[str, Any]) -> None:
    required = ("client_id", "client_secret", "cookie_secret", "redirect_uri", "server_metadata_url")
    if any(not isinstance(auth.get(key), str) or not auth[key].strip() for key in required):
        raise ValueError("Production requires a complete OIDC configuration in Streamlit secrets.")
    if len(auth["cookie_secret"]) < 32 or auth.get("expose_tokens"):
        raise ValueError("Use a strong OIDC cookie secret of at least 32 characters and do not expose tokens.")
    for key in ("redirect_uri", "server_metadata_url"):
        parsed = urlsplit(auth[key])
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("Production OIDC URLs must use HTTPS without embedded credentials.")
    if urlsplit(auth["redirect_uri"]).path != "/oauth2callback":
        raise ValueError("OIDC redirect_uri must use /oauth2callback.")


def identity_allowed(claims: Mapping[str, Any], settings: Settings, client_id: str, now: float | None = None) -> bool:
    current = time.time() if now is None else now
    email = claims.get("email")
    expiry = claims.get("exp")
    issued = claims.get("iat")
    audience = claims.get("aud")
    if claims.get("is_logged_in") is not True or claims.get("email_verified") is not True:
        return False
    if not isinstance(email, str) or email.casefold() not in settings.allowed_emails:
        return False
    if claims.get("iss") != settings.oidc_issuer or not claims.get("sub"):
        return False
    if audience != client_id and not (isinstance(audience, list) and client_id in audience):
        return False
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (expiry, issued)):
        return False
    return expiry > current and issued <= current + 60 and 0 <= current - issued <= settings.session_ttl_seconds


class SessionRateLimiter:
    def __init__(self) -> None:
        self.events: deque[float] = deque()

    def allow(self, limit: int, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        while self.events and self.events[0] <= current - 60:
            self.events.popleft()
        if len(self.events) >= limit:
            return False
        self.events.append(current)
        return True