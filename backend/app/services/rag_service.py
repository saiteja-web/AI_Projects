"""LangChain RAG pipeline — the heart of the project.

Uses the modern LangChain 1.x LCEL (LangChain Expression Language) pattern:
runnables are composed with the pipe operator | instead of the deprecated
RetrievalQA black box.

Two stages:
  INDEXING : PDF -> chunks -> embeddings -> ChromaDB  (build_rag_chain)
  QUERYING : question -> embed -> top-k chunks -> LLM -> answer (query_rag_chain)

Uses a local embedding model (all-MiniLM-L6-v2) and a local Ollama LLM,
so the stack is 100% free and runs entirely in Docker — no external API keys.
"""
import os
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

from app.core.config import settings

# Where ChromaDB persists its vector index (mounted volume in docker-compose)
CHROMA_DIR = os.environ.get("CHROMA_DIR", "/app/chroma_db")

# In-memory registry of active retrieval chains, keyed by session_id.
rag_chains: dict[str, Any] = {}

def _llm() -> ChatOllama:
    """Local LLM using the Ollama container."""
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.3,
    )

def _embeddings() -> HuggingFaceEmbeddings:
    """The local embedding model (downloads ~80MB on first use, then cached)."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def _build_chain(vectordb: Chroma) -> Any:
    """Build an LCEL retrieval chain over a Chroma vector store.

    LCEL pipe composition:
      {context, question} -> prompt -> llm -> string parser
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

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
        | _llm()
        | StrOutputParser()
    )
    return chain, retriever


def build_rag_chain(file_path: str, session_id: str):
    """INDEXING stage: load a PDF, chunk it, embed it, store in ChromaDB,
    and build a retrieval chain ready to answer questions.
    """
    # 1. Load PDF (one Document per page, with page metadata)
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 2. Chunk with overlap so split sentences stay in context
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # 3+4. Embed each chunk and persist to a per-session Chroma collection
    collection_name = f"session_{session_id.replace('-', '_')}"
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings(),
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )

    # 5. Build the LCEL retrieval chain + keep the retriever for citations
    chain, retriever = _build_chain(vectordb)
    rag_chains[session_id] = (chain, retriever)
    return chain


def query_rag_chain(session_id: str, query: str) -> dict[str, Any]:
    """QUERYING stage: answer a question using the session's document.

    Returns {"answer": str, "sources": [page numbers]}.
    """
    cached = rag_chains.get(session_id)
    if cached is None:
        # Reload from persisted Chroma collection (survives restarts)
        collection_name = f"session_{session_id.replace('-', '_')}"
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        chain, retriever = _build_chain(vectordb)
        rag_chains[session_id] = (chain, retriever)
    else:
        chain, retriever = cached

    # 1. Get the answer from the LLM
    answer = chain.invoke(query)

    # 2. Separately fetch the source docs to extract page numbers for citations
    source_docs = retriever.invoke(query)
    source_pages: list[int] = []
    for doc in source_docs:
        page = doc.metadata.get("page")
        if page is not None and page not in source_pages:
            source_pages.append(int(page))

    return {"answer": str(answer).strip(), "sources": source_pages}


def save_uploaded_pdf(upload_file, session_id: str) -> str:
    """Stream an UploadFile to /tmp/uploads/{session_id}_{filename} and return path."""
    upload_dir = "/tmp/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, f"{session_id}_{upload_file.filename}")
    with open(dest, "wb") as buffer:
        buffer.write(upload_file.file.read())
    return dest
