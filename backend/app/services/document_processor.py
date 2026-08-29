"""Section-aware PDF processing for RAG indexing.

Converts a PDF into chunks that know where they came from:
  PDF -> per-page section Documents (heading heuristics)
      -> character chunks (metadata inherited from section doc)
      -> section-prefixed page_content so embeddings and LLM context
         both carry document + section context.

Chunk metadata: {source, page, section, chunk_index}.
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# Headings like "EXPERIENCE" or "Work Experience" — short, no terminal
# punctuation, and every word capitalized (or all caps).
_MAX_HEADING_LEN = 60


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) >= _MAX_HEADING_LEN:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    return stripped.isupper() or stripped.istitle()


def detect_sections(pages: list[Document]) -> list[Document]:
    """Group page text into section Documents using heading heuristics.

    The current section carries forward across page boundaries; a section
    that spans pages produces one Document per page (chunks stay page-true).
    """
    section_docs: list[Document] = []
    current = "Introduction"
    seen_text = False
    for page in pages:
        body: dict[str, list[str]] = {}
        for line in page.page_content.splitlines():
            heading = _is_heading(line)
            if heading and not seen_text and not line.strip().isupper():
                # The first text in a document is usually the title/name
                # block ("John Doe"), which merely looks like a Title Case
                # heading — keep it in the Introduction. An ALL-CAPS first
                # line ("SKILLS") is still read as a section banner.
                heading = False
            if heading:
                current = line.strip()
            if line.strip():
                seen_text = True
            if current not in body:
                body[current] = []
            body[current].append(line)
        # Same heading reappearing later on a page merges its blocks (rare,
        # e.g. repeated table headers) — accepted trade-off.
        for section in body:
            text = "\n".join(body[section]).strip()
            if text:
                section_docs.append(
                    Document(
                        page_content=text,
                        metadata={**page.metadata, "section": section},
                    )
                )
    return section_docs


def process_pdf(file_path: str) -> list[Document]:
    """Load a PDF and return metadata-enriched, section-prefixed chunks."""
    pages = PyPDFLoader(file_path).load()
    section_docs = detect_sections(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(section_docs)

    doc_name = Path(file_path).stem
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.page_content = (
            f"{doc_name} > {chunk.metadata['section']}\n\n{chunk.page_content}"
        )
    return chunks
