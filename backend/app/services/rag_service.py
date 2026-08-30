"""LlamaIndex RAG pipeline — the heart of the project.

Two stages:
  INDEXING : PDF -> section/sentence nodes -> fastembed vectors -> pgvector (HNSW)
             (node_parser.parse_pdf_to_nodes + VectorStoreIndex)
  QUERYING : question -> embed -> top-k session-filtered nodes -> Gemini -> answer
             (retriever + response synthesizer; one retrieval feeds both the
             empty-session guard and synthesis)

Single LLM provider (Gemini); the model is selectable per query.
"""
import os
from typing import Any

from sqlalchemy.engine import make_url
from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings
from app.services.embeddings import FastEmbedEmbedding
from app.services.llm_factory import create_llm, get_default_model
from app.services.node_parser import parse_pdf_to_nodes


class SessionNotIndexedError(LookupError):
    """Raised when a session exists but no document was ever indexed for it."""


SIMILARITY_TOP_K = 10

_QA_PROMPT = PromptTemplate(
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

# In-memory cache of (retriever, synthesizer) per session and per model.
rag_indexes: dict[str, dict[str, tuple[Any, Any]]] = {}


def _configure_llamaindex() -> None:
    """Point LlamaIndex's global embed model at fastembed (idempotent).

    The LLM is NOT set globally — it varies per query via the model picker,
    so it is passed explicitly to the response synthesizer instead.
    """
    LlamaSettings.embed_model = FastEmbedEmbedding(model_name=settings.embedding_model)


def _vector_store() -> PGVectorStore:
    """pgvector store in the app's Postgres: one shared table, HNSW index."""
    url = make_url(settings.database_url)
    return PGVectorStore.from_params(
        database=url.database or "",
        host=url.host or "localhost",
        port=url.port or 5432,
        user=url.username or "",
        password=url.password or "",
        table_name="rag_nodes",
        embed_dim=settings.embed_dim,
        hnsw_kwargs={
            "hnsw_m": settings.hnsw_m,
            "hnsw_ef_construction": settings.hnsw_ef_construction,
            "hnsw_ef_search": settings.hnsw_ef_search,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


def _retriever_for(index: VectorStoreIndex, session_id: str):
    """Top-k retriever scoped to one session via metadata filter."""
    return index.as_retriever(
        similarity_top_k=SIMILARITY_TOP_K,
        filters=MetadataFilters(
            filters=[ExactMatchFilter(key="session_id", value=session_id)]
        ),
    )


def _synthesizer_for(model_id: str):
    return get_response_synthesizer(
        llm=create_llm(model_id, temperature=0.3),
        text_qa_prompt=_QA_PROMPT,
        response_mode="compact",
    )


def build_rag_index(file_path: str, session_id: str, model_id: str | None = None):
    """INDEXING stage: parse the PDF into section/sentence nodes, embed them
    into pgvector, and cache a (retriever, synthesizer) pair for the session.

    Args:
        file_path: Path to the PDF file
        session_id: Unique session identifier
        model_id: LLM model (default model if None)
    """
    model_id = model_id or get_default_model()
    _configure_llamaindex()

    nodes = parse_pdf_to_nodes(file_path, session_id)
    index = VectorStoreIndex(nodes=nodes, vector_store=_vector_store())
    rag_indexes.setdefault(session_id, {})[model_id] = (
        _retriever_for(index, session_id),
        _synthesizer_for(model_id),
    )
    return rag_indexes[session_id][model_id]


def query_rag(
    session_id: str,
    query: str,
    model_id: str | None = None,
) -> dict[str, Any]:
    """QUERYING stage: answer a question using the session's document.

    Args:
        session_id: Unique session identifier
        query: The user's question
        model_id: LLM model (uses default if None)

    Returns {"answer": str, "sources": [{"page": int, "section": str}]}.
    """
    model_id = model_id or get_default_model()
    _configure_llamaindex()

    cached = rag_indexes.get(session_id, {}).get(model_id)
    if cached is None:
        # Reload from persisted pgvector (survives restarts). The embed model
        # comes from the global LlamaSettings that _configure_llamaindex above
        # points at fastembed — reading Settings.embed_model explicitly would
        # lazily resolve an OpenAI default we deliberately don't ship.
        index = VectorStoreIndex.from_vector_store(_vector_store())
        cached = (_retriever_for(index, session_id), _synthesizer_for(model_id))
        rag_indexes.setdefault(session_id, {})[model_id] = cached
    retriever, synthesizer = cached

    # 1. Retrieve top-k nodes for this session (the empty check doubles as
    #    the SessionNotIndexedError guard before any LLM call is spent)
    nodes = retriever.retrieve(query)
    if not nodes:
        raise SessionNotIndexedError(
            f"No document indexed for session {session_id}"
        )

    # 2. Answer from exactly the retrieved nodes
    response = synthesizer.synthesize(query, nodes=nodes)

    # 3. Citations from the nodes actually used (deduped, order-preserving).
    #    NodeWithScore proxies `.metadata` through to its wrapped TextNode.
    sources: list[dict[str, Any]] = []
    for node in nodes:
        source = {
            "page": int(node.metadata.get("page", 0)),
            "section": str(node.metadata.get("section", "unknown")),
        }
        if source not in sources:
            sources.append(source)

    return {"answer": str(response.response or "").strip(), "sources": sources}


def save_uploaded_pdf(upload_file, session_id: str) -> str:
    """Stream an UploadFile to /tmp/uploads/{session_id}_{filename} and return path."""
    upload_dir = "/tmp/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, f"{session_id}_{upload_file.filename}")
    with open(dest, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return dest
