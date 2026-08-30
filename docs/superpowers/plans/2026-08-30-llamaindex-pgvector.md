# LlamaIndex Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LangChain RAG pipeline with idiomatic LlamaIndex: pgvector (HNSW) storage in the existing Postgres, sentence-based section chunking, Gemini via `llama-index-llms-google-genai`, and env-gated LangSmith tracing — with the HTTP API and frontend unchanged.

**Architecture:** A custom `SectionNodeParser(NodeParser)` ports the existing heading heuristics and switches chunking from character-budget to N-sentences-per-chunk. `rag_service` builds a `VectorStoreIndex` over `PGVectorStore` (one shared `rag_nodes` table, session isolation via `ExactMatchFilter` on node metadata) and answers via `retriever.retrieve` + `response_synthesizer.synthesize`. Tracing is a no-op unless `LANGSMITH_TRACING=true`, then OpenLLMetry exports OTLP spans to LangSmith.

**Tech Stack:** Python 3, FastAPI, LlamaIndex (`llama-index-core`, `llama-index-llms-google-genai`, `llama-index-embeddings-fastembed`, `llama-index-vector-stores-postgres`), pgvector, fastembed, pypdf, traceloop-sdk (OpenLLMetry), pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-llamaindex-pgvector-design.md`

**Conventions for every task:**
- All commands run from the repo root unless a step says `cd backend`.
- Use the project venv binaries directly: `backend/.venv/bin/pip`, `backend/.venv/bin/pytest`.
- Tests run with pytest config in `backend/pytest.ini` (asyncio auto mode) — always run from `backend/` so `.env` and paths resolve.
- Test PDF used in manual steps: `/Users/saiteja/Downloads/SriVenkataSivaSaiTejaTankala_Resume.pdf` (exists; the notebook hardcodes this path).
- Postgres creds (compose): user `teja`, password `teja_pass`, db `doc_chat`, port 5432.

---

### Task 1: pgvector infra — compose image, extension, env template

**Files:**
- Modify: `docker-compose.yml` (postgres service `image:` line)
- Modify: `backend/.env.example` (full rewrite of the optional-knobs block)
- Modify: `backend/.env` (append the same optional block; no secrets written)

- [ ] **Step 1: Swap the postgres image in docker-compose.yml**

Change:

```yaml
  postgres:
    image: postgres:16-alpine
```

to:

```yaml
  postgres:
    image: pgvector/pgvector:pg16
```

(One line; everything else in the service stays identical.)

- [ ] **Step 2: Recreate the postgres container on the new image**

```bash
docker compose up -d postgres
docker compose ps postgres
```

Expected: postgres service running, image `pgvector/pgvector:pg16`. Existing `postgres_data` volume is preserved (same major version).

- [ ] **Step 3: Enable the vector extension (idempotent)**

```bash
docker compose exec postgres psql -U teja -d doc_chat -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Expected output: `CREATE EXTENSION` (or a `NOTICE` that it already exists).

Verify: `docker compose exec postgres psql -U teja -d doc_chat -c "\dx vector"` → row for `vector` with version.

- [ ] **Step 4: Rewrite backend/.env.example**

Replace the whole file with:

```bash
# ── Copy this file to .env and fill in real values ─────────────
# cp .env.example .env

# PostgreSQL — "postgres" is the Compose service name (not localhost)
DATABASE_URL=postgresql+asyncpg://teja:teja_pass@postgres:5432/doc_chat

# Gemini API — get a key at https://aistudio.google.com/apikey
GEMINI_API_KEY=

# Default Gemini model (options: gemini-2.5-flash, gemini-2.5-pro, gemini-2.5-flash-lite)
DEFAULT_LLM_MODEL=gemini-2.5-flash

# Local embedding model via fastembed (ONNX, no torch)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Vector dimension of EMBEDDING_MODEL (bge-small-en-v1.5 = 384)
EMBED_DIM=384

# Chunking: a chunk = N whole sentences within a section
SENTENCES_PER_CHUNK=6
SENTENCE_OVERLAP=1

# pgvector HNSW index tuning
HNSW_M=16
HNSW_EF_CONSTRUCTION=64
HNSW_EF_SEARCH=40

# LangSmith tracing (optional) — get a key at https://smith.langchain.com
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=rag-doc-chat

# CORS — the Vite dev server origin
FRONTEND_ORIGIN=http://localhost:5173
```

- [ ] **Step 5: Append the same optional block to backend/.env (no secrets)**

Append to the end of `backend/.env` (do NOT touch existing keys/values, especially `DATABASE_URL` and `GEMINI_API_KEY`):

```bash

# ── LlamaIndex migration knobs (all optional; defaults shown) ──
EMBED_DIM=384
SENTENCES_PER_CHUNK=6
SENTENCE_OVERLAP=1
HNSW_M=16
HNSW_EF_CONSTRUCTION=64
HNSW_EF_SEARCH=40
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=rag-doc-chat
```

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml backend/.env.example
git commit -m "chore: pgvector postgres image + env template for llamaindex knobs"
```

(Do not `git add backend/.env` — it holds the real API key and is gitignored.)

---

### Task 2: Install LlamaIndex dependencies (add-only, alongside LangChain)

**Files:**
- Modify (later, in Task 7): `backend/requirements.txt`

LangChain stays installed until Task 7 so the test suite stays green between tasks. Nothing imports LlamaIndex yet, so no behavior changes.

- [ ] **Step 1: Install the LlamaIndex family + OpenLLMetry into the venv**

```bash
backend/.venv/bin/pip install llama-index-core llama-index-llms-google-genai llama-index-vector-stores-postgres traceloop-sdk
```

(NOTE: `llama-index-embeddings-fastembed` is intentionally NOT installed — it requires Python <3.13 and pins `fastembed<0.2`, incompatible with this py3.13 venv and `fastembed==0.8.0`. A custom wrapper is built in Task 3b instead.)

Expected: installs successfully. If pip reports a version conflict involving `fastembed` or `sqlalchemy`, STOP and report — do not force reinstall.

- [ ] **Step 2: Sanity-check the imports the plan will rely on**

```bash
backend/.venv/bin/python -c "
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import Document, TextNode
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.postgres import PGVectorStore
print('all llama-index imports OK')
"
```

Expected: `all llama-index imports OK`

- [ ] **Step 3: Record resolved versions for Task 7**

```bash
backend/.venv/bin/pip freeze | grep -Ei "^(llama-index|traceloop|fastembed|google-genai|pgvector|psycopg2-binary|opentelemetry-sdk|opentelemetry-exporter-otlp)==" 
```

Save the output — Task 7 Step 2 pins these exact versions.

- [ ] **Step 4: Run the existing suite (must be untouched/green)**

```bash
cd backend && ../.venv 2>/dev/null; .venv/bin/pytest -q; cd ..
```

(If `cd backend` form is awkward, run `cd backend` once and use `.venv/bin/pytest -q`.)

Expected: all existing tests pass — installing new packages changed nothing.

- [ ] **Step 5: No commit (requirements.txt is only rewritten in Task 7)**

Nothing to commit in this task; state is venv-only.

---

### Task 3: Config knobs (embed_dim, chunking, HNSW, LangSmith)

**Files:**
- Modify: `backend/app/core/config.py`
- Test: `backend/tests/test_config.py` (new)

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_config.py`:

```python
"""Regression tests for the LlamaIndex migration config knobs."""
from app.core.config import Settings


def _fresh_settings() -> Settings:
    """Instantiate Settings directly so env-file values don't mask defaults."""
    return Settings(database_url="postgresql+asyncpg://u:p@localhost:5432/db")


def test_rag_knobs_have_defaults():
    s = _fresh_settings()
    assert s.embed_dim == 384
    assert s.sentences_per_chunk == 6
    assert s.sentence_overlap == 1
    assert s.hnsw_m == 16
    assert s.hnsw_ef_construction == 64
    assert s.hnsw_ef_search == 40


def test_langsmith_tracing_off_by_default():
    s = _fresh_settings()
    assert s.langsmith_tracing is False
    assert s.langsmith_project == "rag-doc-chat"
    assert s.langsmith_api_key == ""
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd backend && .venv/bin/pytest tests/test_config.py -q; cd ..
```

Expected: FAIL — `Settings` has no attribute `embed_dim` (pydantic `AttributeError`).

- [ ] **Step 3: Add the fields to Settings**

In `backend/app/core/config.py`, inside the `Settings` class after `default_llm_model` and before `model_config`, add:

```python
    # ── LlamaIndex RAG knobs ─────────────────────────────────
    # Vector dimension of the embedding model (bge-small-en-v1.5 = 384)
    embed_dim: int = 384
    # Chunking: a chunk = N whole sentences within a section
    sentences_per_chunk: int = 6
    sentence_overlap: int = 1
    # pgvector HNSW index tuning
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_ef_search: int = 40

    # ── LangSmith tracing (optional; OTLP via OpenLLMetry) ──
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-doc-chat"
    langsmith_tracing: bool = False
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd backend && .venv/bin/pytest tests/test_config.py -q; cd ..
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/config.py backend/tests/test_config.py
git commit -m "feat: config knobs for llamaindex rag (chunking, hnsw, langsmith)"
```

---

### Task 3b: Custom FastEmbed embedding wrapper (TDD)

**Files:**
- Create: `backend/app/services/embeddings.py`
- Test: `backend/tests/test_embeddings.py` (new)

`llama-index-embeddings-fastembed` cannot be installed on this py3.13 venv with `fastembed==0.8.0` (see Task 2 note). This wrapper implements LlamaIndex's `BaseEmbedding` over `fastembed` directly — the same pattern LlamaIndex integrations use internally.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_embeddings.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
cd backend && .venv/bin/pytest tests/test_embeddings.py -q; cd ..
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.embeddings'`.

- [ ] **Step 3: Implement embeddings.py**

Create `backend/app/services/embeddings.py`:

```python
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
        return list(next(self._model.embed([text])))

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [list(vec) for vec in self._model.embed(texts)]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
```

(`BaseEmbedding` is a pydantic model; the fastembed client is held in a `PrivateAttr`.)

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd backend && .venv/bin/pytest tests/test_embeddings.py -q; cd ..
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/embeddings.py backend/tests/test_embeddings.py
git commit -m "feat: custom fastembed embedding wrapper for llamaindex"
```

---

### Task 4: SectionNodeParser — sentence-based section chunking (TDD)

**Files:**
- Create: `backend/app/services/node_parser.py`
- Test: `backend/tests/test_node_parser.py` (new; `document_processor.py` is untouched in this task and deleted in Task 6)

The parser owns: pypdf loading (`load_pdf_pages`), section detection (`detect_sections`, ported verbatim from `document_processor.py`), sentence splitting, N-sentences grouping, and node construction (metadata + section prefix).

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_node_parser.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
cd backend && .venv/bin/pytest tests/test_node_parser.py -q; cd ..
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.node_parser'`.

- [ ] **Step 3: Implement node_parser.py**

Create `backend/app/services/node_parser.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd backend && .venv/bin/pytest tests/test_node_parser.py -q; cd ..
```

Expected: all pass (~16 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/node_parser.py backend/tests/test_node_parser.py
git commit -m "feat: SectionNodeParser with sentence-based section chunking"
```

---

### Task 5: llm_factory → GoogleGenAI

**Files:**
- Modify: `backend/app/services/llm_factory.py` (import + `create_llm` body)
- Modify: `backend/tests/test_llm_factory.py:23-27` (model-name assertion)

- [ ] **Step 1: Update the failing test**

In `backend/tests/test_llm_factory.py`, replace `test_create_llm_builds_gemini_model` (lines 23–27) with:

```python
def test_create_llm_builds_gemini_model(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    llm = llm_factory.create_llm("gemini-2.5-flash", temperature=0.3)
    # llama-index-llms-google-genai stores the model id verbatim on .model
    assert llm.model == "gemini-2.5-flash"
    assert llm.temperature == 0.3
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd backend && .venv/bin/pytest tests/test_llm_factory.py -q; cd ..
```

Expected: FAIL — `AttributeError: module 'app.services.llm_factory' has no attribute ...` / `TypeError` from `ChatGoogleGenerativeAI` not having `.model` semantics. (The old import still works; the assertion changes.)

- [ ] **Step 3: Swap the implementation**

In `backend/app/services/llm_factory.py`:

Replace line 8:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
```

with:

```python
from llama_index.llms.google_genai import GoogleGenAI
```

Replace the return type hint and body of `create_llm` (lines 53–73) with:

```python
def create_llm(model_id: str, temperature: float = 0.3) -> GoogleGenAI:
    """Create a Gemini chat model instance.

    Args:
        model_id: Model identifier (must be in AVAILABLE_MODELS)
        temperature: Sampling temperature (0.0 to 1.0)

    Raises:
        ValueError: If model_id is unknown or GEMINI_API_KEY is not configured
    """
    if get_model_by_id(model_id) is None:
        raise ValueError(
            f"Unknown model: {model_id}. Available: {[m.id for m in AVAILABLE_MODELS]}"
        )
    if not settings.gemini_api_key:
        raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env")
    return GoogleGenAI(
        model=model_id,
        temperature=temperature,
        api_key=settings.gemini_api_key,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd backend && .venv/bin/pytest tests/test_llm_factory.py -q; cd ..
```

Expected: all pass (7 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/llm_factory.py backend/tests/test_llm_factory.py
git commit -m "feat: llm factory returns llama-index GoogleGenAI (gemini)"
```

---

### Task 6: rag_service rewrite + router call sites (the big swap)

**Files:**
- Modify: `backend/app/services/rag_service.py` (full rewrite)
- Modify: `backend/app/routers/upload.py:38-42` (function rename + comment)
- Modify: `backend/app/routers/chat.py:38` (function rename)
- Modify: `backend/tests/test_rag_service.py` (full rewrite)
- Modify: `backend/tests/test_chat.py:51,84` (patch target rename)
- Delete: `backend/app/services/document_processor.py`
- Delete: `backend/tests/test_document_processor.py`

- [ ] **Step 1: Rewrite the failing unit tests first**

Replace `backend/tests/test_rag_service.py` with:

```python
"""Unit tests for rag_service guards (no DB, no model downloads, no API)."""
from types import SimpleNamespace

import pytest

from app.services import rag_service
from app.services.rag_service import SessionNotIndexedError


class _FakeRetriever:
    def __init__(self, nodes):
        self._nodes = nodes

    def retrieve(self, query):
        return self._nodes


class _FakeIndex:
    def __init__(self, retriever):
        self._retriever = retriever

    def as_retriever(self, **kwargs):
        return self._retriever


class _FakeNode:
    def __init__(self, page, section):
        self.metadata = {"page": page, "section": section}


def _patch_pipeline(monkeypatch, retrieved_nodes):
    """Stub everything between rag_service and the outside world."""
    monkeypatch.setattr(rag_service, "_configure_llamaindex", lambda: None)
    monkeypatch.setattr(rag_service, "_vector_store", lambda: object())
    monkeypatch.setattr(
        rag_service.VectorStoreIndex,
        "from_vector_store",
        lambda store, embed_model=None: _FakeIndex(_FakeRetriever(retrieved_nodes)),
    )
    monkeypatch.setattr(
        rag_service,
        "_synthesizer_for",
        lambda model_id: SimpleNamespace(
            synthesize=lambda query, nodes: SimpleNamespace(response=" 42 days ")
        ),
    )


def test_query_never_indexed_session_raises(monkeypatch):
    _patch_pipeline(monkeypatch, retrieved_nodes=[])

    with pytest.raises(SessionNotIndexedError):
        rag_service.query_rag("never-indexed-session", "hello")


def test_query_returns_answer_and_deduped_sources(monkeypatch):
    nodes = [
        _FakeNode(1, "EXPERIENCE"),
        _FakeNode(1, "EXPERIENCE"),
        _FakeNode(2, "SKILLS"),
    ]
    _patch_pipeline(monkeypatch, retrieved_nodes=nodes)

    result = rag_service.query_rag("s1", "how much leave?", model_id="gemini-2.5-flash")

    assert result["answer"] == "42 days"
    assert result["sources"] == [
        {"page": 1, "section": "EXPERIENCE"},
        {"page": 2, "section": "SKILLS"},
    ]
```

- [ ] **Step 2: Run to verify failure**

```bash
cd backend && .venv/bin/pytest tests/test_rag_service.py -q; cd ..
```

Expected: FAIL — `AttributeError: ... has no attribute 'query_rag'`.

- [ ] **Step 3: Rewrite rag_service.py**

Replace the entire contents of `backend/app/services/rag_service.py` with:

```python
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
        # Reload from persisted pgvector (survives restarts)
        index = VectorStoreIndex.from_vector_store(
            _vector_store(), embed_model=LlamaSettings.embed_model
        )
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

    # 3. Citations from the nodes actually used (deduped, order-preserving)
    sources: list[dict[str, Any]] = []
    for node in nodes:
        source = {
            "page": int(node.node.metadata.get("page", 0)),
            "section": str(node.node.metadata.get("section", "unknown")),
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
```

- [ ] **Step 4: Update the two router call sites**

`backend/app/routers/upload.py` — replace lines 38–42:

```python
    # 4. Build the RAG chain (loads + chunks + embeds + indexes into ChromaDB)
    #    This is synchronous/heavy — brute-force OK for dev. Production would use
    #    a background task (FastAPI BackgroundTasks) so the HTTP call returns fast.
    try:
        rag_service.build_rag_chain(file_path, session_id)
```

with:

```python
    # 4. Build the RAG index (parse + embed + store in pgvector)
    #    This is synchronous/heavy — brute-force OK for dev. Production would use
    #    a background task (FastAPI BackgroundTasks) so the HTTP call returns fast.
    try:
        rag_service.build_rag_index(file_path, session_id)
```

`backend/app/routers/chat.py` — replace line 38:

```python
        result = rag_service.query_rag_chain(
```

with:

```python
        result = rag_service.query_rag(
```

- [ ] **Step 5: Update test_chat.py patch targets**

In `backend/tests/test_chat.py`, replace both occurrences of:

```python
        "app.routers.chat.rag_service.query_rag_chain",
```

with:

```python
        "app.routers.chat.rag_service.query_rag",
```

Also update the docstring in line 4–5 from `mocks query_rag_chain` to `mocks query_rag`.

- [ ] **Step 6: Delete the dead LangChain document processor**

```bash
git rm backend/app/services/document_processor.py backend/tests/test_document_processor.py
```

- [ ] **Step 7: Run the full suite**

```bash
cd backend && .venv/bin/pytest -q; cd ..
```

Expected: all tests pass (test_rag_service, test_chat, test_llm_factory, test_node_parser, test_config, test_upload, test_history). `grep -rn "langchain\|chroma" backend/app backend/tests --include="*.py"` returns nothing.

- [ ] **Step 8: Commit**

```bash
git add backend/app/services/rag_service.py backend/app/routers/upload.py backend/app/routers/chat.py backend/tests/test_rag_service.py backend/tests/test_chat.py
git commit -m "feat: rag service on llamaindex (pgvector+hnsw, session filters) and gemini query engine"
```

---

### Task 7: Remove LangChain from requirements and pin LlamaIndex versions

**Files:**
- Modify: `backend/requirements.txt` (full rewrite)

- [ ] **Step 1: Confirm nothing imports langchain anymore**

```bash
grep -rn "langchain" backend/app backend/tests --include="*.py"; echo "exit=$?"
```

Expected: no matches (`exit=1` from grep means "nothing found").

- [ ] **Step 2: Rewrite backend/requirements.txt**

Replace the whole file with the versions captured in Task 2 Step 3 (the `pip freeze` output). The structure must be:

```
# ── Web framework ──────────────────────────────────────────────
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.9          # required by FastAPI for file uploads

# ── Config / env ───────────────────────────────────────────────
pydantic==2.10.3
pydantic-settings==2.11.0
python-dotenv==1.0.1             # notebook reads backend/.env via load_dotenv

# ── Database (async + pgvector sync driver) ────────────────────
sqlalchemy==2.0.35
greenlet==3.5.5                  # required by SQLAlchemy async engine
asyncpg==0.30.0                  # async PostgreSQL driver
psycopg2-binary==2.9.10          # sync driver used by llama-index PGVectorStore

# ── RAG pipeline (LlamaIndex) ─────────────────────────────────
llama-index-core==<PIN from pip freeze>
llama-index-llms-google-genai==<PIN>
llama-index-vector-stores-postgres==<PIN>
fastembed==<PIN — currently 0.8.0>
pypdf==<PIN — currently 5.0.1>
pgvector==<PIN if present in pip freeze>

# ── Observability (LangSmith via OpenLLMetry/OTel) ────────────
traceloop-sdk==<PIN>

# ── Notebook / testing ─────────────────────────────────────────
ipykernel==6.29.5                # Jupyter kernel for the experiments notebook
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2                    # async client for testing FastAPI
```

Every `<PIN>` must be replaced with the real resolved version from Task 2 Step 3 (`backend/.venv/bin/pip freeze | grep -Ei "^(llama-index|traceloop|fastembed|pypdf|google-genai|pgvector)=="`). A `<PIN>` left in the file is a plan failure.

- [ ] **Step 3: Verify the pinned set installs cleanly in a throwaway check**

```bash
backend/.venv/bin/pip check
```

Expected: `No broken requirements found.` (If conflicts appear, adjust pins to the versions pip actually resolved and re-run.)

- [ ] **Step 4: Uninstall the LangChain family and confirm the suite is still green**

```bash
backend/.venv/bin/pip uninstall -y langchain langchain-core langchain-chroma langchain-community langchain-google-genai langchain-text-splitters langgraph langgraph-checkpoint langgraph-sdk langsmith 2>/dev/null
cd backend && .venv/bin/pytest -q; cd ..
```

Expected: suite green with LangChain packages gone. (Uninstalling extras like langgraph/langsmith is harmless if not present — `2>/dev/null` covers that.)

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: swap langchain stack for llamaindex family in requirements"
```

---

### Task 8: LangSmith tracing (env-gated OpenLLMetry)

**Files:**
- Create: `backend/app/core/tracing.py`
- Modify: `backend/app/main.py` (lifespan wires `setup_tracing`)
- Test: `backend/tests/test_tracing.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_tracing.py`:

```python
"""Tests for env-gated LangSmith tracing setup (offline, no OTel SDK)."""
import os
import sys
from types import SimpleNamespace

from app.core import tracing
from app.core.config import settings


def test_tracing_disabled_is_a_noop(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", False)
    # No env touched, no traceloop import attempted (nothing stubbed → would
    # blow up if setup_tracing tried to import it).
    assert tracing.setup_tracing() is None


def test_tracing_enabled_configures_traceloop(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_api_key", "test-key")
    monkeypatch.setattr(settings, "langsmith_project", "test-project")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)

    class _FakeTraceloop:
        def __init__(self):
            self.calls = {}

        def init(self, **kwargs):
            self.calls.update(kwargs)

    fake = _FakeTraceloop()
    monkeypatch.setitem(
        sys.modules, "traceloop", SimpleNamespace(sdk=SimpleNamespace(Traceloop=fake))
    )
    monkeypatch.setitem(
        sys.modules, "traceloop.sdk", SimpleNamespace(Traceloop=fake)
    )

    tracing.setup_tracing()

    assert fake.calls["app_name"] == "test-project"
    assert (
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
        == "https://api.smith.langchain.com/otel/v1/traces"
    )
    assert "x-api-key=test-key" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]
    assert "Langsmith-Project=test-project" in os.environ["OTEL_EXPORTER_OTLP_HEADERS"]


def test_tracing_without_api_key_is_skipped(monkeypatch):
    monkeypatch.setattr(settings, "langsmith_tracing", True)
    monkeypatch.setattr(settings, "langsmith_api_key", "")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert tracing.setup_tracing() is None
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in os.environ
```

- [ ] **Step 2: Run to verify failure**

```bash
cd backend && .venv/bin/pytest tests/test_tracing.py -q; cd ..
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.core.tracing'`.

- [ ] **Step 3: Implement tracing.py**

Create `backend/app/core/tracing.py`:

```python
"""Env-gated LangSmith tracing via OpenLLMetry (OpenTelemetry OTLP).

LangSmith's first-class integrations are LangChain-shaped; for LlamaIndex
the supported path is OpenTelemetry: OpenLLMetry (traceloop-sdk) auto-
instruments LlamaIndex + the Gemini SDK and exports OTLP spans to LangSmith's
OTel endpoint, where they appear as retrieval spans (retrieved nodes, scores)
and generation spans (prompt, tokens, latency) in LANGSMITH_PROJECT.

Disabled by default (LANGSMITH_TRACING=false): zero overhead, no imports.
"""
import logging
import os

from app.core.config import settings

logger = logging.getLogger(__name__)

_LANGSMITH_OTLP_ENDPOINT = "https://api.smith.langchain.com/otel/v1/traces"


def setup_tracing() -> None:
    """Configure OTLP tracing to LangSmith. No-op unless LANGSMITH_TRACING=true."""
    if not settings.langsmith_tracing:
        return
    if not settings.langsmith_api_key:
        logger.warning(
            "LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty — tracing disabled"
        )
        return

    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", _LANGSMITH_OTLP_ENDPOINT)
    os.environ.setdefault(
        "OTEL_EXPORTER_OTLP_HEADERS",
        f"x-api-key={settings.langsmith_api_key},"
        f"Langsmith-Project={settings.langsmith_project}",
    )

    # Imported lazily: traceloop-sdk is heavy and only needed when tracing.
    from traceloop.sdk import Traceloop

    Traceloop.init(app_name=settings.langsmith_project)
    logger.info(
        "LangSmith tracing enabled (project=%s, endpoint=%s)",
        settings.langsmith_project,
        _LANGSMITH_OTLP_ENDPOINT,
    )
```

- [ ] **Step 4: Wire into the app lifespan**

In `backend/app/main.py`, add the import after `from app.core.database import ...` (line 15):

```python
from app.core.tracing import setup_tracing
```

and make the lifespan body start with the tracing call:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up optional LangSmith tracing, then create DB tables on startup."""
    setup_tracing()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd backend && .venv/bin/pytest tests/test_tracing.py -q && .venv/bin/pytest -q; cd ..
```

Expected: tracing tests pass; full suite green.

- [ ] **Step 6: Commit**

```bash
git add backend/app/core/tracing.py backend/app/main.py backend/tests/test_tracing.py
git commit -m "feat: env-gated langsmith tracing via openllmetry otlp"
```

---

### Task 9: Rebuild the experiments notebook for LlamaIndex

**Files:**
- Modify: `backend/notebooks/rag_pipeline_experiments.ipynb` (full rebuild, one stage per cell)

- [ ] **Step 1: Rebuild the notebook**

Use the NotebookEdit tool on `backend/notebooks/rag_pipeline_experiments.ipynb`: delete all existing cells, then insert the following cells in order.

Cell 1 (markdown):

```markdown
# RAG pipeline experiments — LlamaIndex + pgvector + Gemini

One stage per cell, with printed intermediates. Runtime: `backend/.venv`
(select it as the kernel). Stage order:

0. Setup
1. LOAD — pypdf pages
2. PARSE — `SectionNodeParser` (sections + sentence chunks)
3. EMBED — fastembed vectors
4. STORE — `PGVectorStore` (HNSW)
5. RETRIEVE — session-filtered retriever with scores
6. ANSWER — Gemini response synthesizer
7. PLAYGROUND — tweak knobs and watch retrieval change
```

Cell 2 (code) — setup:

```python
# 0. SETUP — path, env, key check
import os
import sys

sys.path.insert(0, "..")
from dotenv import load_dotenv

load_dotenv("../.env")
assert os.getenv("GEMINI_API_KEY"), "Set GEMINI_API_KEY in backend/.env"
print("setup ok")
```

Cell 3 (code) — load:

```python
# 1. LOAD — pypdf gives one page per document; metadata carries source + page
from app.services.node_parser import load_pdf_pages

PDF_PATH = "/Users/saiteja/Downloads/SriVenkataSivaSaiTejaTankala_Resume.pdf"
pages = load_pdf_pages(PDF_PATH)

print(f"{len(pages)} pages")
print("page metadata:", pages[0].metadata)
print(pages[0].text[:300])
```

Cell 4 (code) — parse:

```python
# 2. PARSE — sections from heading heuristics, chunks = N whole sentences
from app.services.node_parser import SectionNodeParser

parser = SectionNodeParser(
    session_id="notebook", sentences_per_chunk=6, sentence_overlap=1
)
nodes = parser.get_nodes_from_documents(pages)

print(f"{len(nodes)} nodes")
print("metadata:", nodes[0].metadata)
print(nodes[0].get_content()[:300])
```

Cell 5 (code) — embed:

```python
# 3. EMBED — local ONNX vectors (downloads ~130MB on first use, then cached)
from app.services.embeddings import FastEmbedEmbedding

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
vec = embed_model.get_text_embedding(nodes[0].get_content())

print(f"{len(vec)} dims; first 5: {vec[:5]}")
```

Cell 6 (code) — store:

```python
# 4. STORE — pgvector table 'rag_nodes' with an HNSW index (cosine)
from sqlalchemy.engine import make_url
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore

from app.core.config import settings as app_settings

Settings.embed_model = embed_model

url = make_url(app_settings.database_url)
store = PGVectorStore.from_params(
    database=url.database,
    host=url.host,
    port=url.port,
    user=url.username,
    password=url.password,
    table_name="rag_nodes",
    embed_dim=app_settings.embed_dim,
    hnsw_kwargs={
        "hnsw_m": app_settings.hnsw_m,
        "hnsw_ef_construction": app_settings.hnsw_ef_construction,
        "hnsw_ef_search": app_settings.hnsw_ef_search,
        "hnsw_dist_method": "vector_cosine_ops",
    },
)
index = VectorStoreIndex(nodes=nodes, vector_store=store)
print("indexed into rag_nodes (HNSW)")
```

Cell 7 (code) — retrieve:

```python
# 5. RETRIEVE — metadata filter scopes results to this session's nodes
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=MetadataFilters(
        filters=[ExactMatchFilter(key="session_id", value="notebook")]
    ),
)
question = "What experience does the candidate have?"  # <- change me
hits = retriever.retrieve(question)

for h in hits[:5]:
    print(round(h.score, 3), f"p.{h.node.metadata['page']}", h.node.metadata["section"])
```

Cell 8 (code) — answer:

```python
# 6. ANSWER — Gemini synthesizes from exactly the retrieved nodes
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.google_genai import GoogleGenAI

QA_PROMPT = PromptTemplate(
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context_str}\n\nQuestion: {query_str}\n\nAnswer:"
)
synth = get_response_synthesizer(
    llm=GoogleGenAI(
        model="gemini-2.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.3,
    ),
    text_qa_prompt=QA_PROMPT,
)
response = synth.synthesize(question, nodes=hits)
print(response.response)
```

Cell 9 (markdown) — playground:

```markdown
## 7. PLAYGROUND

Things to tweak and re-run cells 4–8:

- `sentences_per_chunk` (try 3 vs 10) — chunk granularity vs context size
- `similarity_top_k` (try 3 vs 20) — recall vs noise
- `hnsw_ef_search` (set `app_settings.hnsw_ef_search = 80` before cell 6) —
  HNSW search width vs recall/speed
- `sentence_overlap` (try 0 vs 2) — context continuity across chunk boundaries

Watch how the scores and the cited `p.N / section` change.
```

- [ ] **Step 2: Sanity-check the notebook JSON parses**

```bash
backend/.venv/bin/python -c "import nbformat; nb=nbformat.read('backend/notebooks/rag_pipeline_experiments.ipynb', as_version=4); nbformat.validate(nb); print(len(nb.cells), 'cells, valid')"
```

Expected: `9 cells, valid`

- [ ] **Step 3: Commit**

```bash
git add backend/notebooks/rag_pipeline_experiments.ipynb
git commit -m "docs: rebuild experiments notebook for llamaindex/pgvector pipeline"
```

---

### Task 10: Write the tradeoffs document

**Files:**
- Create: `docs/llamaindex-vs-langchain-tradeoffs.md`

- [ ] **Step 1: Collect the measured numbers**

```bash
git diff --stat 62b6cc3..HEAD -- backend/app | tail -1
```

(62b6cc3 is the last commit before the migration. Record the insertions/deletions line.)

- [ ] **Step 2: Write the document**

Create `docs/llamaindex-vs-langchain-tradeoffs.md` — keep the structure below and replace `<N>` placeholders with the measured numbers from Step 1:

```markdown
# LlamaIndex vs LangChain — observed tradeoffs

Notes from converting this repo's RAG pipeline (PDF → section/sentence nodes →
pgvector HNSW → Gemini) from LangChain 1.x to LlamaIndex, [date].

## What changed concretely

| Concern | LangChain (before) | LlamaIndex (after) |
|---|---|---|
| Orchestration | LCEL pipe: `retriever \| format \| prompt \| llm \| parser` | retriever + response synthesizer composed by hand |
| Document model | `Document{page_content, metadata}` | `TextNode{text, metadata}` (+ node relationships) |
| Custom chunking | compose `PyPDFLoader` + `RecursiveCharacterTextSplitter` | subclass `NodeParser` (first-class hook, `_parse_nodes`) |
| Vector store | `langchain-chroma` (local files) | `PGVectorStore` (HNSW in the existing Postgres) |
| LLM | `ChatGoogleGenerativeAI` | `GoogleGenAI` (same 3-model registry) |
| Prompt | f-string template | `PromptTemplate` with `{context_str}`/`{query_str}` conventions |
| Answer flow | `chain.invoke(query)` + separate `retriever.invoke(query)` (retrieval ran twice) | `retriever.retrieve(query)` then `synthesizer.synthesize(query, nodes=...)` (retrieval once; citations = exactly the used nodes) |
| Observability | LangSmith native callbacks | OTLP via OpenLLMetry → LangSmith OTel endpoint |

Code size: backend RAG services went from ~N to ~M lines (git 62b6cc3..HEAD).

## Tradeoffs we actually noticed

1. **Explicitness vs integration depth.** The LCEL pipe made every step visible
   as a runnable stage. LlamaIndex bundles retrieval + synthesis into a query
   engine — fewer moving parts to wire, but you accept its defaults unless you
   drop to the retriever/synthesizer layer (which we did, to avoid double
   retrieval and to guard empty sessions before spending an LLM call).
2. **Chunking is a first-class extension point.** In LangChain, custom chunking
   means chaining loaders + splitters + ad-hoc metadata surgery. In LlamaIndex,
   `NodeParser` is the designed seam — our section/sentence logic is one class.
3. **Observability has a detour.** LangSmith is LangChain-native; for
   LlamaIndex you route OpenTelemetry spans through OpenLLMetry into LangSmith's
   OTLP endpoint. Works, but it is one more moving part (and env-gated).
4. **Citations got simpler.** `Response.source_nodes` / retrieved nodes carry
   scores + metadata, so `p.N · section` citations come from the exact nodes
   the answer used — no second retrieval pass like the LCEL version needed.
5. **Vector store choice is where the leverage is.** Moving Chroma→pgvector
   bought HNSW tunables (`m`, `ef_construction`, `ef_search`) and dropped a
   storage engine — the retrieval knobs now live in env vars, not code.
6. **Ecosystem fit.** LangChain feels broader (agents, tools, 100+ loaders);
   LlamaIndex feels deeper on the ingest→index→retrieve path. For a RAG app
   like this one, the depth is the better fit.

## When to pick which

- LlamaIndex: document-centric apps, custom ingest/chunking, retrieval quality
  tuning, vector-store experimentation.
- LangChain: agent/tool orchestration, when the team already speaks LCEL, or
  when LangSmith-native tracing matters more than retrieval depth.
```

- [ ] **Step 3: Commit**

```bash
git add docs/llamaindex-vs-langchain-tradeoffs.md
git commit -m "docs: llamaindex vs langchain tradeoffs from the migration"
```

---

### Task 11: End-to-end verification (manual, needs running Postgres + GEMINI_API_KEY)

**Files:** none (verification only)

- [ ] **Step 1: Start the stack's DB and confirm the extension**

```bash
docker compose up -d postgres
docker compose exec postgres psql -U teja -d doc_chat -c "\dx vector"
```

Expected: vector extension listed. If the backend `.env` DATABASE_URL points at `localhost` (native run), that still resolves — compose publishes 5432.

- [ ] **Step 2: Start the backend natively**

```bash
cd backend && .venv/bin/uvicorn app.main:app --port 8001
```

Expected: startup logs clean; `GET http://localhost:8001/health` → `{"status": "ok"}`.

- [ ] **Step 3: Upload + chat through the API**

```bash
curl -s -F "file=@/Users/saiteja/Downloads/SriVenkataSivaSaiTejaTankala_Resume.pdf" http://localhost:8001/api/upload
```

Expected: `{"session_id": "<id>", "filename": "...", "status": "ready"}`. First call downloads the fastembed model (~130 MB) — allow time.

```bash
curl -s -X POST http://localhost:8001/api/chat -H 'Content-Type: application/json' \
  -d '{"session_id": "<id-from-upload>", "query": "What experience does the candidate have with Python?"}'
```

Expected: 200 with `answer` + `sources` where every source is `{"page": int, "section": str}`.

- [ ] **Step 4: Restart-survival (pgvector lazy reload)**

Stop the backend (Ctrl-C), start it again, re-run the chat curl with the same session_id.

Expected: 200 answer again (index reloaded from pgvector, no re-upload).

- [ ] **Step 5: LangSmith tracing round-trip (optional, needs LANGSMITH_API_KEY)**

Set `LANGSMITH_TRACING=true` and a real key in `backend/.env`, restart the backend, run one chat call, open https://smith.langchain.com → project `rag-doc-chat`.

Expected: one trace per chat call containing retrieval spans (retrieved nodes/scores) and Gemini generation spans (prompt, tokens, latency).

If Traceloop's LlamaIndex instrumentation produces empty/broken traces: implement the spec's fallback — manual `opentelemetry.sdk.TracerProvider` + `OTLPSpanExporter` (endpoint + headers already env-driven) registered as LlamaIndex's global handler via `llama_index.core.global_handler`/dispatcher span handler — and note it in the tradeoffs doc.

- [ ] **Step 6: HNSW knob experiment**

Set `HNSW_EF_SEARCH=80` in `backend/.env`, restart the backend, re-run the same chat query.

Expected: retrieval still works; similarity scores in the LangSmith trace (or add a debug log) differ from the ef_search=40 run — the knob is live.

- [ ] **Step 7: Frontend regression sweep (if frontend is running)**

`cd frontend && npm run dev` → open http://localhost:5173 → upload the PDF → chat → model picker shows the 3 Gemini models → citations render `p.N · section`.

Expected: identical UX to before the migration (frontend was not touched).

- [ ] **Step 8: Final commit checkpoint**

```bash
git status   # confirm clean tree; push decision is the user's
```

---

## Self-review checklist (completed during plan writing)

- Spec coverage: infra (T1), deps (T2, T7), config knobs (T3), node parser + sentence chunking (T4), GoogleGenAI factory (T5), rag_service rewrite + routers + tests (T6), LangSmith tracing (T8), notebook (T9), tradeoffs doc (T10), verification incl. HNSW knob + restart + tracing (T11). Frontend: intentionally untouched. `chroma_db/`: left in place per spec.
- No placeholders in code steps; the only execution-time values are pip-pinned versions (exact command provided) and measured LOC for the tradeoffs doc (exact command provided).
- Type consistency: `build_rag_index`/`query_rag` names match across rag_service, routers, and tests; `SectionNodeParser(session_id=..., sentences_per_chunk=..., sentence_overlap=...)` matches in parser and notebook; `rag_indexes` cache shape `(retriever, synthesizer)` consistent; test monkeypatch stubs match real signatures (`as_retriever(**kwargs)`, `retrieve(query)`, `synthesize(query, nodes=...)`).
