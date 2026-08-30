"""Tests for env-gated LangSmith tracing setup (offline, no OTel SDK)."""
import os
import sys
from types import SimpleNamespace

from app.core import tracing
from app.core.config import settings


def test_tracing_disabled_is_a_noop(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", False)
    # No env touched, no traceloop import attempted (nothing stubbed → would
    # blow up if setup_tracing tried to import it).
    assert tracing.setup_tracing() is None


def test_tracing_enabled_configures_traceloop(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_api_key", "test-key")
    monkeypatch.setattr(settings, "langsmith_project", "test-project")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)

    class _FakeTraceloop:
        def __init__(self):
            self.calls = {}

        def init(self, **kwargs):
            self.calls.update(kwargs)

    fake = _FakeTraceloop()
    monkeypatch.setitem(
        sys.modules, "traceloop", SimpleNamespace(sdk=SimpleNamespace(Traceloop=fake))
    )
    monkeypatch.setitem(
        sys.modules, "traceloop.sdk", SimpleNamespace(Traceloop=fake)
    )

    tracing.setup_tracing()

    assert fake.calls["app_name"] == "test-project"
    assert (
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        == "https://api.smith.langchain.com/otel/v1/traces"
    )
    assert "x-api-key=test-key" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
    assert "Langsmith-Project=test-project" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]


def test_tracing_without_api_key_is_skipped(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_api_key", "")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert tracing.setup_tracing() is None
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ
