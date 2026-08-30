"""Tests for env-gated LangSmith tracing setup (offline, no OTel SDK)."""
import sys
from types import SimpleNamespace

from app.core import tracing
from app.core.config import settings


def _stub_traceloop(monkeypatch, init_impl):
    """Install fake traceloop modules whose Traceloop.init runs init_impl."""
    calls = {}

    def init(**kwargs):
        calls.update(kwargs)
        init_impl(**kwargs)

    fake = SimpleNamespace(init=init)
    monkeypatch.setitem(
        sys.modules, "traceloop", SimpleNamespace(sdk=SimpleNamespace(Traceloop=fake))
    )
    monkeypatch.setitem(
        sys.modules, "traceloop.sdk", SimpleNamespace(Traceloop=fake)
    )
    return calls


def test_tracing_disabled_is_a_noop(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", False)
    # No traceloop stub → would blow up if setup_tracing tried to import it.
    assert tracing.setup_tracing() is None


def test_tracing_enabled_configures_traceloop(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_api_key", "test-key")
    monkeypatch.setattr(settings, "langsmith_project", "test-project")

    calls = _stub_traceloop(monkeypatch, lambda **kw: None)
    tracing.setup_tracing()

    assert calls["app_name"] == "test-project"
    assert calls["api_endpoint"] == "https://api.smith.langchain.com/otel/v1/traces"
    assert calls["headers"]["x-api-key"] == "test-key"
    assert calls["headers"]["Langsmith-Project"] == "test-project"


def test_tracing_without_api_key_is_skipped(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_api_key", "")

    assert tracing.setup_tracing() is None


def test_traceloop_failure_degrades_without_raising(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_api_key", "test-key")

    def _boom(**kwargs):
        raise RuntimeError("exporter construction failed")

    _stub_traceloop(monkeypatch, _boom)

    # Observability must degrade — never take app startup down.
    assert tracing.setup_tracing() is None
