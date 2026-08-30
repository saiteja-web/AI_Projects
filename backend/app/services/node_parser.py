"""Section-aware PDF parsing for LlamaIndex RAG indexing.

Converts a PDF into LlamaIndex TextNodes that know where they came from:
  PDF -> per-page Documents (pypdf)
      -> section grouping (heading heuristics; sections carry across pages)
      -> sentence chunks: N whole sentences per chunk within a section,
         never cut mid-sentence and not constrained by character budget
      -> TextNodes with metadata {session_id, source, page, section,
         chunk_index} and section-prefixed text so embeddings AND the LLM
         context carry document + section context.

Node text: "doc > section\n\n<chunk sentences>"
"""
import re
from pathlib import Path

from pypdf import PdfReader
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, Document, TextNode

from app.core.config import settings

# Headings like "EXPERIENCE" or "Work Experience" — short, no terminal
# punctuation, and every word capitalized (or all caps).
_MAX_HEADING_LEN = 60
# Sentence boundary: terminal punctuation followed by whitespace.
# Bullet lines without terminal punctuation merge into the following
# sentence — accepted trade-off (resume bullets are short).
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


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
        for line in (page.text or "").splitlines():
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
                        text=text,
                        metadata={**page.metadata, "section": section},
                    )
                )
    return section_docs


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at terminal punctuation + whitespace."""
    return [s for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def group_sentences(sentences: list[str], size: int, overlap: int) -> list[str]:
    """Group consecutive sentences into space-joined chunks.

    Chunks after the first start `overlap` sentences before the previous
    chunk ended, so context carries across chunk boundaries.
    """
    if not sentences:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(sentences):
        end = min(start + size, len(sentences))
        chunks.append(" ".join(sentences[start:end]))
        if end == len(sentences):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_pdf_pages(pdf_path: str) -> list[Document]:
    """Load a PDF as one Document per page with {source: file stem, page}.

    Page numbers are 0-based, matching the citations the app has always
    rendered ("p.0" is the first page).
    """
    reader = PdfReader(pdf_path)
    doc_name = Path(pdf_path).stem
    return [
        Document(text=page.extract_text() or "", metadata={"source": doc_name, "page": i})
        for i, page in enumerate(reader.pages)
    ]


class SectionNodeParser(NodeParser):
    """Parse page Documents into section-aware, sentence-grouped TextNodes."""

    session_id: str
    sentences_per_chunk: int = 6
    sentence_overlap: int = 1

    @classmethod
    def class_name(cls) -> str:
        return "SectionNodeParser"

    def _parse_nodes(
        self,
        nodes: list[BaseNode],
        show_progress: bool = False,
        **kwargs: object,
    ) -> list[BaseNode]:
        # One call over ALL pages so the current section carries across
        # page boundaries (same as the original detect_sections contract).
        parsed: list[BaseNode] = []
        for section_doc in detect_sections(list(nodes)):
            source = str(section_doc.metadata["source"])
            section = str(section_doc.metadata["section"])
            sentences = split_sentences(section_doc.text)
            for chunk_text in group_sentences(
                sentences, self.sentences_per_chunk, self.sentence_overlap
            ):
                parsed.append(
                    TextNode(
                        text=f"{source} > {section}\n\n{chunk_text}",
                        metadata={
                            "session_id": self.session_id,
                            "source": source,
                            "page": int(section_doc.metadata.get("page", 0)),
                            "section": section,
                            "chunk_index": len(parsed),
                        },
                    )
                )
        return parsed


def parse_pdf_to_nodes(pdf_path: str, session_id: str) -> list[TextNode]:
    """Full INDEXING parse: PDF pages -> section/sentence TextNodes."""
    pages = load_pdf_pages(pdf_path)
    parser = SectionNodeParser(
        session_id=session_id,
        sentences_per_chunk=settings.sentences_per_chunk,
        sentence_overlap=settings.sentence_overlap,
    )
    return parser.get_nodes_from_documents(pages)
