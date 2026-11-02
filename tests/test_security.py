from dataclasses import replace
from unittest.mock import patch

import pytest

from support_assistant.config import Settings
from support_assistant.security import SessionRateLimiter, identity_allowed, validate_oidc_config


@pytest.fixture
def production():
    return Settings(environment="production", username="synthetic-importer", password="test-only",
                    reader_username="synthetic-reader", reader_password="test-only",
                    allowed_emails=("agent@solari.example",), oidc_issuer="https://idp.example.invalid",
                    resolution_age_days=180)


@pytest.fixture
def claims():
    return {"is_logged_in": True, "email_verified": True, "email": "agent@solari.example",
            "iss": "https://idp.example.invalid", "sub": "synthetic-agent", "aud": "test-client", "iat": 900, "exp": 2000}


def test_allowed_identity(production, claims):
    assert identity_allowed(claims, production, "test-client", now=1000)


@pytest.mark.parametrize("change", [
    {"is_logged_in": False}, {"email_verified": False}, {"email_verified": "true"},
    {"email": "outsider@solari.example"}, {"email": None}, {"exp": 999}, {"exp": None},
    {"iat": 10000}, {"iat": -5000}, {"exp": float("nan")}, {"aud": "other-client"},
    {"iss": "https://other-idp.example.invalid"}, {"sub": ""},
])
def test_untrusted_or_expired_claims_denied(production, claims, change):
    assert not identity_allowed({**claims, **change}, production, "test-client", now=1000)


@pytest.mark.parametrize("change", [
    {"reader_username": "synthetic-importer"}, {"reader_username": "", "reader_password": ""},
    {"allowed_emails": ()}, {"oidc_issuer": ""}, {"resolution_age_days": 0},
    {"url": "http://db.example.invalid:5984"},
])
def test_production_configuration_fails_closed(production, change):
    with pytest.raises(ValueError):
        replace(production, **change)


def test_reader_uses_distinct_credentials(production):
    reader = production.for_reader()
    assert reader.username == "synthetic-reader"
    assert reader.role == "reader"
    assert "test-only" not in repr(reader)


def test_production_web_process_does_not_require_ingestion_secret(monkeypatch):
    values = {"APP_ENV": "production", "COUCHDB_URL": "http://127.0.0.1:5984",
              "COUCHDB_READ_USER": "synthetic-reader", "COUCHDB_READ_PASSWORD": "test-only",
              "AUTH_ALLOWED_EMAILS": "agent@solari.example", "APP_OIDC_ISSUER": "https://idp.example.invalid",
              "RESOLUTION_MAX_AGE_DAYS": "180"}
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("COUCHDB_USER", raising=False)
    monkeypatch.delenv("COUCHDB_PASSWORD", raising=False)
    with patch("support_assistant.config.load_dotenv"):
        settings = Settings.from_env(role="reader")
    assert settings.username == "synthetic-reader"
    assert settings.role == "reader"


def test_session_rate_limit_expires():
    limiter = SessionRateLimiter()
    assert limiter.allow(2, now=0)
    assert limiter.allow(2, now=1)
    assert not limiter.allow(2, now=2)
    assert limiter.allow(2, now=61)


@pytest.mark.parametrize("config", [
    {}, {"cookie_secret": "short"},
    {"client_id": "synthetic", "client_secret": "synthetic", "cookie_secret": "x" * 32,
     "redirect_uri": "http://support.example.invalid/oauth2callback", "server_metadata_url": "https://idp.example.invalid"},
])
def test_incomplete_oidc_configuration_denied(config):
    with pytest.raises(ValueError):
        validate_oidc_config(config)


def test_valid_oidc_configuration():
    validate_oidc_config({"client_id": "synthetic", "client_secret": "synthetic", "cookie_secret": "x" * 32,
                          "redirect_uri": "https://support.example.invalid/oauth2callback",
                          "server_metadata_url": "https://idp.example.invalid/.well-known/openid-configuration"})