"""Tests for the Gemini-only model factory."""
import pytest

from app.core.config import settings
from app.services import llm_factory


def test_available_models_are_gemini_only():
    ids = [m.id for m in llm_factory.AVAILABLE_MODELS]
    assert ids == ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]


def test_model_info_has_no_provider_field():
    model = llm_factory.AVAILABLE_MODELS[0]
    assert not hasattr(model, "provider")


def test_get_default_model_returns_model_string():
    assert llm_factory.get_default_model() == settings.default_llm_model
    assert isinstance(llm_factory.get_default_model(), str)


def test_create_llm_builds_gemini_model(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    llm = llm_factory.create_llm("gemini-2.5-flash", temperature=0.3)
    # langchain-google-genai 3.2.0 stores the model as "models/<id>" on .model
    assert llm.model.endswith("gemini-2.5-flash")


def test_create_llm_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        llm_factory.create_llm("gpt-4o")


def test_get_model_by_id_finds_gemini():
    model = llm_factory.get_model_by_id("gemini-2.5-pro")
    assert model is not None
    assert model.name == "Gemini 2.5 Pro"


def test_create_llm_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        llm_factory.create_llm("gemini-2.5-flash", temperature=0.3)
