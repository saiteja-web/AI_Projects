"""Unit tests for rag_service guards (no DB, no model downloads)."""
import pytest
from chromadb.errors import NotFoundError

from app.services import rag_service
from app.services.rag_service import SessionNotIndexedError


class NotFoundErrorStub(NotFoundError):
    """Raised by the fake Chroma to mimic a missing backing collection."""

    pass


def test_query_never_indexed_session_raises(monkeypatch, tmp_path):
    def fake_chroma(**kwargs):
        raise NotFoundErrorStub()

    monkeypatch.setattr(rag_service, "Chroma", fake_chroma)
    monkeypatch.setattr(rag_service, "CHROMA_DIR", str(tmp_path))
    # Keep the unit test offline: skip the fastembed model load entirely.
    monkeypatch.setattr(rag_service, "_embeddings", lambda: None)

    with pytest.raises(SessionNotIndexedError):
        rag_service.query_rag_chain("never-indexed-session", "hello")
