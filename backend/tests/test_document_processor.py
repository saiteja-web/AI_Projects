"""Tests for section detection and metadata-enriched chunking."""
from types import SimpleNamespace

from langchain_core.documents import Document

from app.services.document_processor import (
    _is_heading,
    detect_sections,
    process_pdf,
)


class TestIsHeading:
    def test_uppercase(self):
        assert _is_heading("EXPERIENCE") is True

    def test_title_case(self):
        assert _is_heading("Work Experience") is True

    def test_sentence_rejected(self):
        assert _is_heading("Built REST APIs using Node.js and FastAPI.") is False

    def test_empty_rejected(self):
        assert _is_heading("   ") is False

    def test_long_line_rejected(self):
        assert _is_heading("a" * 80) is False

    def test_lowercase_rejected(self):
        assert _is_heading("built rest apis") is False

    def test_numbered_heading(self):
        assert _is_heading("1. Experience") is True

    def test_date_range_rejected(self):
        assert _is_heading("2021 - 2023") is False

    def test_comma_skill_line_becomes_heading_is_accepted_tradeoff(self):
        assert _is_heading("REACT, PYTHON, SQL") is True


class TestDetectSections:
    def test_splits_page_into_sections(self):
        page = Document(
            page_content=(
                "John Doe\n"
                "EXPERIENCE\n"
                "Built APIs at Acme Corp.\n"
                "EDUCATION\n"
                "B.Tech in ECE.\n"
            ),
            metadata={"source": "resume.pdf", "page": 0},
        )
        docs = detect_sections([page])
        sections = [d.metadata["section"] for d in docs]
        assert sections == ["Introduction", "EXPERIENCE", "EDUCATION"]

    def test_carries_section_across_pages(self):
        page1 = Document(
            page_content="SKILLS\nPython and FastAPI.\n",
            metadata={"source": "r.pdf", "page": 0},
        )
        page2 = Document(
            page_content="More skill details here.\n",
            metadata={"source": "r.pdf", "page": 1},
        )
        docs = detect_sections([page1, page2])
        assert docs[-1].metadata["section"] == "SKILLS"
        assert docs[-1].metadata["page"] == 1

    def test_metadata_includes_source_and_page(self):
        page = Document(
            page_content="Some body text.\n",
            metadata={"source": "r.pdf", "page": 3},
        )
        docs = detect_sections([page])
        assert docs[0].metadata["source"] == "r.pdf"
        assert docs[0].metadata["page"] == 3
        assert docs[0].metadata["section"] == "Introduction"


class TestProcessPdf:
    def _patch_loader(self, monkeypatch, pages):
        monkeypatch.setattr(
            "app.services.document_processor.PyPDFLoader",
            lambda path: SimpleNamespace(load=lambda: pages),
        )

    def test_chunks_carry_section_metadata_and_prefix(self, monkeypatch):
        self._patch_loader(
            monkeypatch,
            [
                Document(
                    page_content="EXPERIENCE\n" + "Built APIs at Acme Corp. " * 40,
                    metadata={"source": "resume.pdf", "page": 0},
                )
            ],
        )
        chunks = process_pdf("resume.pdf")
        assert len(chunks) > 1
        assert all(c.metadata["section"] == "EXPERIENCE" for c in chunks)
        assert all(
            c.page_content.startswith("resume > EXPERIENCE\n\n") for c in chunks
        )

    def test_chunk_index_is_sequential(self, monkeypatch):
        self._patch_loader(
            monkeypatch,
            [
                Document(
                    page_content="SKILLS\n" + "Python, FastAPI, React. " * 60,
                    metadata={"source": "r.pdf", "page": 0},
                )
            ],
        )
        chunks = process_pdf("r.pdf")
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_no_sections_yields_no_chunks(self, monkeypatch):
        self._patch_loader(monkeypatch, [])
        assert process_pdf("empty.pdf") == []
