import pytest


@pytest.fixture(autouse=True)
def lexical_mode_by_default(request, monkeypatch):
    # Unit tests must not depend on whether a local model has been downloaded.
    # Tests that need semantic search inject an encoder or are marked "semantic".
    if "semantic" not in request.keywords:
        monkeypatch.setattr("support_assistant.matching.semantic_model_source", lambda: None)
        monkeypatch.setattr("support_assistant.service.semantic_model_source", lambda: None)
