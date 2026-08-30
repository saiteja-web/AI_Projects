"""Custom fastembed → LlamaIndex embedding wrapper.

The llama-index-embeddings-fastembed integration package cannot be used on
this venv (it requires Python <3.13 and pins fastembed<0.2, while the project
pins fastembed==0.8.0). This implements LlamaIndex's BaseEmbedding over
fastembed directly — the same shape its official integrations use.
"""
from typing import List

from fastembed import TextEmbedding
from pydantic import PrivateAttr

from llama_index.core.base.embeddings.base import BaseEmbedding


class FastEmbedEmbedding(BaseEmbedding):
    """Local ONNX embeddings (downloads the model on first use, then cached)."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    _model: TextEmbedding = PrivateAttr()

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self._model = TextEmbedding(model_name=model_name)

    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return next(self._model.embed([text])).tolist()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Public batch API (removed from BaseEmbedding in llama_index >= 0.11)."""
        return self._get_text_embeddings(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
