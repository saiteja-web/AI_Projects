"""LangChain RAG pipeline — the heart of the project.

Uses LangChain 1.x LCEL (pipe-composed runnables).

Two stages:
  INDEXING : PDF -> section-aware chunks -> fastembed vectors -> ChromaDB
             (document_processor.process_pdf + build_rag_chain)
  QUERYING : question -> embed -> top-k chunks -> Gemini -> answer
             (query_rag_chain)

Single LLM provider (Gemini); the model is selectable per query.
"""
import os
from typing import Any

from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import FastEmbedEmbeddings  # langchain-fastembed pkg doesn't exist; community wrapper over fastembed

from app.core.config import settings
from app.services.document_processor import process_pdf
from app.services.llm_factory import create_llm, get_default_model


class SessionNotIndexedError(LookupError):
    """Raised when a session exists but no document was ever indexed for it."""


# Where ChromaDB persists its vector index. Relative default works both
# natively (cwd backend/) and in the container (cwd /app = mounted volume).
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")

# In-memory registry of active retrieval chains, keyed by session_id.
# Each entry maps model_id -> (chain, retriever)
rag_chains: dict[str, dict[str, tuple[Any, Any]]] = {}


def _collection_for(session_id: str) -> str:
    """Chroma collection name for a session (must match build and query)."""
    return f"session_{session_id.replace('-', '_')}"


def _llm(model_id: str):
    """Create the Gemini LLM via the factory."""
    return create_llm(model_id, temperature=0.3)


def _embeddings() -> FastEmbedEmbeddings:
    """Local ONNX embeddings (downloads ~130MB on first use, then cached)."""
    return FastEmbedEmbeddings(model_name=settings.embedding_model)


def _build_chain(vectordb: Chroma, model_id: str) -> Any:
    """Build an LCEL retrieval chain over a Chroma vector store.

    LCEL pipe composition:
      {context, question} -> prompt -> llm -> string parser
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    prompt = PromptTemplate.from_template(
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | _llm(model_id)
        | StrOutputParser()
    )
    return chain, retriever


def build_rag_chain(file_path: str, session_id: str, model_id: str | None = None):
    """INDEXING stage: load a PDF into section-aware chunks, embed, store in
    ChromaDB, and build a retrieval chain ready to answer questions.

    Args:
        file_path: Path to the PDF file
        session_id: Unique session identifier
        model_id: LLM model (default model if None)
    """
    model_id = model_id or get_default_model()

    # 1+2. Load, detect sections, chunk with metadata
    chunks = process_pdf(file_path)

    # 3. Embed each chunk and persist to a per-session Chroma collection
    collection_name = _collection_for(session_id)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )

    # 4. Build the LCEL retrieval chain + keep the retriever for citations
    chain, retriever = _build_chain(vectordb, model_id)

    rag_chains.setdefault(session_id, {})[model_id] = (chain, retriever)
    return chain


def query_rag_chain(
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

    cached = rag_chains.get(session_id, {}).get(model_id)
    if cached is None:
        # Reload from persisted Chroma collection (survives restarts)
        collection_name = _collection_for(session_id)
        try:
            vectordb = Chroma(
                collection_name=collection_name,
                embedding_function=_embeddings(),
                persist_directory=CHROMA_DIR,
                create_collection_if_not_exists=False,
            )
        except NotFoundError:
            raise SessionNotIndexedError(
                f"No document indexed for session {session_id}"
            ) from None
        chain, retriever = _build_chain(vectordb, model_id)
        rag_chains.setdefault(session_id, {})[model_id] = (chain, retriever)
    else:
        chain, retriever = cached

    # 1. Get the answer from the LLM
    answer = chain.invoke(query)

    # 2. Separately fetch source docs for page + section citations
    source_docs = retriever.invoke(query)
    sources: list[dict[str, Any]] = []
    for doc in source_docs:
        source = {
            "page": int(doc.metadata.get("page", 0)),
            "section": str(doc.metadata.get("section", "unknown")),
        }
        if source not in sources:
            sources.append(source)

    return {"answer": str(answer).strip(), "sources": sources}


def save_uploaded_pdf(upload_file, session_id: str) -> str:
    """Stream an UploadFile to /tmp/uploads/{session_id}_{filename} and return path."""
    upload_dir = "/tmp/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, f"{session_id}_{upload_file.filename}")
    with open(dest, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return dest
