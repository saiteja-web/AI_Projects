"""Tests for section detection and sentence-based node parsing."""
import pytest
from llama_index.core.schema import Document

from app.services import node_parser
from app.services.node_parser import (
    SectionNodeParser,
    _is_heading,
    detect_sections,
    group_sentences,
    split_sentences,
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


class TestSplitSentences:
    def test_basic_split(self):
        assert split_sentences("One sentence. Two sentences! Three?") == [
            "One sentence.",
            "Two sentences!",
            "Three?",
        ]

    def test_filters_blank_fragments(self):
        assert split_sentences("  First.   \n  Second. ") == ["First.", "Second."]

    def test_text_without_terminal_punctuation_is_one_sentence(self):
        assert split_sentences("no punctuation here") == ["no punctuation here"]


class TestGroupSentences:
    def test_exact_groups_without_overlap(self):
        s = [f"S{i}." for i in range(7)]
        assert group_sentences(s, size=3, overlap=0) == [
            "S0. S1. S2.",
            "S3. S4. S5.",
            "S6.",
        ]

    def test_overlap_carries_last_sentence_forward(self):
        s = [f"S{i}." for i in range(5)]
        assert group_sentences(s, size=3, overlap=1) == [
            "S0. S1. S2.",
            "S2. S3. S4.",
        ]

    def test_fewer_sentences_than_size_is_single_chunk(self):
        assert group_sentences(["Only one."], size=6, overlap=1) == ["Only one."]


class TestDetectSections:
    def test_splits_page_into_sections(self):
        page = Document(
            text=(
                "John Doe\n"
                "EXPERIENCE\n"
                "Built APIs at Acme Corp.\n"
                "EDUCATION\n"
                "B.Tech in ECE.\n"
            ),
            metadata={"source": "resume", "page": 0},
        )
        docs = detect_sections([page])
        sections = [d.metadata["section"] for d in docs]
        assert sections == ["Introduction", "EXPERIENCE", "EDUCATION"]

    def test_carries_section_across_pages(self):
        page1 = Document(text="SKILLS\nPython and FastAPI.\n", metadata={"source": "r", "page": 0})
        page2 = Document(text="More skill details here.\n", metadata={"source": "r", "page": 1})
        docs = detect_sections([page1, page2])
        assert docs[-1].metadata["section"] == "SKILLS"
        assert docs[-1].metadata["page"] == 1

    def test_metadata_includes_source_and_page(self):
        page = Document(text="Some body text.\n", metadata={"source": "r", "page": 3})
        docs = detect_sections([page])
        assert docs[0].metadata["source"] == "r"
        assert docs[0].metadata["page"] == 3
        assert docs[0].metadata["section"] == "Introduction"


class TestSectionNodeParser:
    def _parser(self, **overrides) -> SectionNodeParser:
        defaults = dict(
            session_id="s1", sentences_per_chunk=2, sentence_overlap=0
        )
        defaults.update(overrides)
        return SectionNodeParser(**defaults)

    def test_nodes_carry_section_metadata_and_prefix(self):
        pages = [
            Document(
                text="EXPERIENCE\nBuilt APIs at Acme Corp. Shipped features. Led the team.",
                metadata={"source": "resume", "page": 0},
            )
        ]
        nodes = self._parser().get_nodes_from_documents(pages)
        assert len(nodes) == 2
        assert all(n.metadata["section"] == "EXPERIENCE" for n in nodes)
        assert all(n.get_content().startswith("resume > EXPERIENCE\n\n") for n in nodes)
        assert "Built APIs at Acme Corp. Shipped features." in nodes[0].get_content()
        assert "Led the team." in nodes[1].get_content()

    def test_session_id_injected_and_chunk_index_sequential(self):
        pages = [
            Document(
                text="SKILLS\nPython, FastAPI, React. Strong testing. Systems design. Docker.",
                metadata={"source": "r", "page": 0},
            )
        ]
        nodes = self._parser(session_id="sess-9").get_nodes_from_documents(pages)
        assert len(nodes) == 2
        assert all(n.metadata["session_id"] == "sess-9" for n in nodes)
        assert [n.metadata["chunk_index"] for n in nodes] == [0, 1]
        assert all(n.metadata["source"] == "r" for n in nodes)
        assert all(n.metadata["page"] == 0 for n in nodes)


class TestParsePdfToNodes:
    def test_loads_pages_and_injects_session_id(self, monkeypatch):
        pages = [
            Document(
                text="SKILLS\nPython, FastAPI, React. Testing. Design.",
                metadata={"source": "r", "page": 0},
            )
        ]
        captured = {}

        real_parser = node_parser.SectionNodeParser
        monkeypatch.setattr(node_parser, "load_pdf_pages", lambda p: pages)
        monkeypatch.setattr(
            node_parser,
            "SectionNodeParser",
            lambda **kw: (captured.update(kw), real_parser(**kw))[1],
        )

        nodes = node_parser.parse_pdf_to_nodes("r.pdf", "sess-1")

        assert captured["session_id"] == "sess-1"
        assert nodes
        assert all(n.metadata["session_id"] == "sess-1" for n in nodes)
        assert [n.metadata["chunk_index"] for n in nodes] == list(range(len(nodes)))


class TestLoadPdfPages:
    def test_real_pdf_pages_have_source_and_zero_based_page(self):
        # Tiny synthetic PDF via pypdf writer keeps this offline and fast.
        from pypdf import PdfWriter
        import io

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)
        with open("/tmp/_np_test.pdf", "wb") as f:
            f.write(buf.read())

        pages = node_parser.load_pdf_pages("/tmp/_np_test.pdf")
        assert len(pages) == 1
        assert pages[0].metadata["source"] == "_np_test"
        assert pages[0].metadata["page"] == 0
