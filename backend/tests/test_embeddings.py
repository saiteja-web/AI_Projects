"""Tests for the custom fastembed→LlamaIndex embedding wrapper (offline)."""
import pytest

from app.services import embeddings as embeddings_module
from app.services.embeddings import FastEmbedEmbedding


class _FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding so tests never download a model."""

    def __init__(self, model_name):
        self.model_name = model_name

    def embed(self, texts):
        return iter([[0.1, 0.2, 0.3] for _ in texts])


@pytest.fixture
def fake_fastembed(monkeypatch):
    monkeypatch.setattr(embeddings_module, "TextEmbedding", _FakeTextEmbedding)


def test_wraps_fastembed_with_configured_model(fake_fastembed):
    emb = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    assert emb.model_name == "BAAI/bge-small-en-v1.5"
    assert emb._model.model_name == "BAAI/bge-small-en-v1.5"


def test_query_and_text_embeddings_return_vectors(fake_fastembed):
    emb = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    assert emb.get_query_embedding("hello") == [0.1, 0.2, 0.3]
    assert emb.get_text_embedding("hello") == [0.1, 0.2, 0.3]


def test_batch_text_embeddings(fake_fastembed):
    emb = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    assert emb.get_text_embeddings(["a", "b"]) == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
