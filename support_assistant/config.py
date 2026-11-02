import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    url: str = "http://127.0.0.1:5984"
    database: str = "solari_support_tickets"
    incoming_database: str = "solari_incoming_tickets"
    feedback_database: str = "solari_search_feedback"
    username: str = ""
    password: str = field(default="", repr=False)
    environment: str = "development"
    reader_username: str = ""
    reader_password: str = field(default="", repr=False)
    role: str = "ingestion"
    allowed_emails: tuple[str, ...] = ()
    oidc_issuer: str = ""
    resolution_age_days: int = 0
    max_tickets: int = 50000
    min_score: float = 0.30
    search_limit_per_minute: int = 20
    session_ttl_seconds: int = 3600
    triage_routing_file: Path = ROOT / "config" / "triage_routing.json"

    def __post_init__(self) -> None:
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("COUCHDB_URL must be an HTTP or HTTPS server URL.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use separate CouchDB credential variables, not credentials or query parameters in the URL.")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.database):
            raise ValueError("COUCHDB_DATABASE must start with a lowercase letter and contain only lowercase letters, digits, hyphens, or underscores (max 64 characters).")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.incoming_database) or self.incoming_database == self.database:
            raise ValueError("COUCHDB_INCOMING_DATABASE must be a valid, separate database name.")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.feedback_database) or self.feedback_database in {self.database, self.incoming_database}:
            raise ValueError("COUCHDB_FEEDBACK_DATABASE must be a valid, separate database name.")
        if bool(self.username) != bool(self.password):
            raise ValueError("Set both COUCHDB_USER and COUCHDB_PASSWORD, or leave both unset.")
        if bool(self.reader_username) != bool(self.reader_password):
            raise ValueError("Set both COUCHDB_READ_USER and COUCHDB_READ_PASSWORD, or leave both unset.")
        if self.environment not in {"development", "production"}:
            raise ValueError("APP_ENV must be development or production.")
        if self.role not in {"ingestion", "reader"}:
            raise ValueError("Settings role must be ingestion or reader.")
        limits = ((self.max_tickets, 1, 1000000), (self.resolution_age_days, 0, 3650),
                  (self.search_limit_per_minute, 1, 1000), (self.session_ttl_seconds, 60, 86400))
        if any(isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper
               for value, lower, upper in limits):
            raise ValueError("Application limits must be integers within the documented bounds.")
        if isinstance(self.min_score, bool) or not isinstance(self.min_score, (int, float)) or not 0 < self.min_score <= 1:
            raise ValueError("MIN_SIMILARITY must be greater than zero and at most one.")
        if self.environment == "production":
            if not self.username or not self.reader_username or (self.role == "ingestion" and self.username == self.reader_username):
                raise ValueError("Production requires authenticated, distinct ingestion and read credentials.")
            if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} and parsed.scheme != "https":
                raise ValueError("Production CouchDB connections outside loopback require HTTPS.")
            issuer = urlsplit(self.oidc_issuer)
            if issuer.scheme != "https" or not issuer.hostname or issuer.username or issuer.query or issuer.fragment:
                raise ValueError("Production requires an HTTPS APP_OIDC_ISSUER.")
            if not self.allowed_emails or any("@" not in email for email in self.allowed_emails):
                raise ValueError("Production requires an explicit AUTH_ALLOWED_EMAILS allowlist.")
            if self.resolution_age_days == 0:
                raise ValueError("Production requires a positive RESOLUTION_MAX_AGE_DAYS.")

    def for_reader(self) -> "Settings":
        if self.reader_username:
            return replace(self, username=self.reader_username, password=self.reader_password, role="reader")
        return replace(self, role="reader")

    def for_incoming(self) -> "Settings":
        reader = self.for_reader()
        return replace(reader, database=self.incoming_database, incoming_database=self.database)

    def for_feedback(self, admin: bool = False) -> "Settings":
        base = self if admin else self.for_reader()
        return replace(base, database=self.feedback_database, feedback_database=self.database)

    @classmethod
    def from_env(cls, role: str = "ingestion") -> "Settings":
        load_dotenv(ROOT / ".env", override=False)
        environment = os.getenv("APP_ENV", "development")
        reader_username = os.getenv("COUCHDB_READ_USER", "")
        reader_password = os.getenv("COUCHDB_READ_PASSWORD", "")
        if role == "reader" and (reader_username or environment == "production"):
            username, password = reader_username, reader_password
        else:
            username, password = os.getenv("COUCHDB_USER", ""), os.getenv("COUCHDB_PASSWORD", "")
        return cls(
            url=os.getenv("COUCHDB_URL", "http://127.0.0.1:5984").rstrip("/"),
            database=os.getenv("COUCHDB_DATABASE", "solari_support_tickets"),
            incoming_database=os.getenv("COUCHDB_INCOMING_DATABASE", "solari_incoming_tickets"),
            feedback_database=os.getenv("COUCHDB_FEEDBACK_DATABASE", "solari_search_feedback"),
            username=username,
            password=password,
            environment=environment,
            reader_username=reader_username,
            reader_password=reader_password,
            role=role,
            allowed_emails=tuple(email.strip().casefold() for email in os.getenv("AUTH_ALLOWED_EMAILS", "").split(",") if email.strip()),
            oidc_issuer=os.getenv("APP_OIDC_ISSUER", ""),
            resolution_age_days=int(os.getenv("RESOLUTION_MAX_AGE_DAYS", "180" if environment == "production" else "0")),
            max_tickets=int(os.getenv("MAX_HISTORY_TICKETS", "50000")),
            min_score=float(os.getenv("MIN_SIMILARITY", "0.30")),
            search_limit_per_minute=int(os.getenv("SEARCH_LIMIT_PER_MINUTE", "20")),
            session_ttl_seconds=int(os.getenv("AUTH_SESSION_TTL_SECONDS", "3600")),
            triage_routing_file=Path(os.getenv("TRIAGE_ROUTING_FILE", "").strip() or ROOT / "config" / "triage_routing.json"),
        )