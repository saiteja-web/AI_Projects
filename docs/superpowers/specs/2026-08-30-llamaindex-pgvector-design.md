# LlamaIndex migration: LangChain out, pgvector + HNSW in, LangSmith tracing

Date: 2026-08-30
Status: Approved (pending implementation plan)

## Goal

Convert the RAG pipeline from LangChain to LlamaIndex, as a learning-driven rewrite:

1. Replace the entire LangChain stack (LCEL, PyPDFLoader, RecursiveCharacterTextSplitter, langchain-chroma, langchain-google-genai) with idiomatic LlamaIndex
2. Replace ChromaDB with pgvector on the existing Postgres, using an HNSW index with tunable parameters
3. Switch chunking from character-budget to sentence-budget (N complete sentences per chunk within a section)
4. Add LangSmith observability via OpenTelemetry tracing
5. Produce a tradeoffs document capturing LangChain-vs-LlamaIndex differences observed during the migration

Motivating constraints: user is learning LlamaIndex; wants HNSW access (Pinecone hides ANN internals, so it was rejected); Postgres runs via docker-compose and stays the only container; native backend runs use `backend/.venv` (no Docker for the app until the upgrade is done); frontend and HTTP API must not change.

## Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| Scope | Full pipeline migration; LangChain removed from requirements entirely |
| Vector store | pgvector on existing Postgres (`doc_chat`), HNSW index, cosine distance |
| Why not Pinecone | ANN algorithm is opaque — no HNSW access or tuning |
| Chunking unit | N complete sentences per chunk (default 6, overlap 1), grouped within a section — not character-constrained |
| LLM | Gemini only, via `llama-index-llms-gemini`; same 3-model registry |
| Embeddings | fastembed `BAAI/bge-small-en-v1.5` (384-dim) via `llama-index-embeddings-fastembed` |
| Observability | LangSmith, OTLP ingest (`https://api.smith.langchain.com/otel`), env-gated |
| API contract | Unchanged: `{answer, sources: [{page, section}]}`; frontend untouched |
| Postgres image | Swap `postgres:16-alpine` → `pgvector/pgvector:pg16` (drop-in, same volume) |

## Architecture

```
INDEXING: PDF → pypdf pages
              → SectionNodeParser (ported heading heuristics → section text + metadata)
              → sentence split → group N sentences per chunk (overlap 1)
              → TextNodes  {session_id, source, page, section, chunk_index}
              → FastEmbedEmbedding (BAAI/bge-small-en-v1.5, 384-dim)
              → VectorStoreIndex → PGVectorStore (HNSW, cosine)

QUERYING: question → VectorIndexRetriever (similarity_top_k=10,
                     MetadataFilters: session_id == this session)
                     → response synthesizer w/ existing prompt ("answer ONLY from context…")
                     → Gemini → Response
                     → {answer, sources: [{page, section}] (deduped)}
```

The HTTP API surface is byte-identical to today: same request/response schemas, same routers, same frontend. Only the internals and service function names change.

## Backend changes

### `app/services/node_parser.py` (new; replaces `document_processor.py`)

- Custom `SectionNodeParser(NodeParser)` — the idiomatic LlamaIndex extension point for custom chunking.
- Heading detection ported verbatim from `document_processor.py`: `_is_heading` heuristics (short, no terminal punctuation, UPPER/Title case), "Introduction" default, first-title-block guard.
- Loads PDF with `pypdf` directly (no `PyPDFLoader`).
- Sentence splitting: pragmatic regex-based splitter (`.`/`!`/`?` followed by whitespace) — no heavy NLP dependency. Exact implementation decided in the plan.
- Chunks = N consecutive sentences per section (default 6, overlap 1 sentence), configurable via Settings.
- Per node: metadata `{session_id, source, page, section, chunk_index}`; `page_content` prefixed with `f"{doc_name} > {section}\n\n"` (embeddings and LLM context keep section context, as today).

### `app/services/llm_factory.py`

- `ModelInfo`, `AVAILABLE_MODELS`, `get_model_by_id`, `get_default_model` unchanged.
- `create_llm` returns `Gemini` (from `llama_index.llms.gemini`) with the same `(model_id, temperature, api_key)` semantics.

### `app/services/rag_service.py` (rewritten)

- `build_rag_chain` → `build_rag_index(file_path, session_id, model_id=None)`:
  1. `SectionNodeParser` produces nodes (with `session_id` injected)
  2. `VectorStoreIndex(nodes=..., vector_store=pg_store, embed_model=...)` upserts into pgvector
  3. Query engine cached in `rag_indexes[session_id][model_id]` (same in-memory cache shape as today)
- `query_rag_chain` → `query_rag(session_id, query, model_id=None)`:
  1. `VectorStoreIndex.from_vector_store(pg_store, embed_model=...)`
  2. `index.as_retriever(similarity_top_k=10, filters=MetadataFilters([ExactMatchFilter("session_id", ...)]))`
  3. Query engine with the existing prompt template (answer ONLY from context / say you don't know) as `text_qa_template`
  4. `SessionNotIndexedError` raised when the filtered retrieval returns zero nodes
  5. Sources: deduped `{page: int, section: str}` from source-node metadata — identical shape
- `SessionNotIndexedError` kept; `save_uploaded_pdf` unchanged; module keeps its name and `rag_indexes` cache (renamed from `rag_chains`).
- DB credentials parsed from `settings.database_url` via SQLAlchemy `make_url` — no new env vars.

### `app/core/config.py`

Adds (all with defaults, env-overridable free via pydantic-settings):

```python
langsmith_api_key: str = ""
langsmith_project: str = "rag-doc-chat"
langsmith_tracing: bool = False
sentences_per_chunk: int = 6
sentence_overlap: int = 1
hnsw_m: int = 16
hnsw_ef_construction: int = 64
hnsw_ef_search: int = 40
```

Everything existing stays (database_url, embedding_model, gemini_api_key, default_llm_model, frontend_origin).

### `app/core/tracing.py` (new)

- Env-gated by `settings.langsmith_tracing`.
- OpenLLMetry (`traceloop-sdk`): `Traceloop.init(app_name=...)` auto-instruments LlamaIndex + Gemini SDK; OTLP exporter pointed at `https://api.smith.langchain.com/otel` (append `/v1/traces` as needed) with headers `x-api-key=<langsmith_api_key>` and project name header.
- Called at app startup (`main.py` lifespan).
- Fallback (decided during implementation if Traceloop's LlamaIndex instrumentation misbehaves): manual OTel `TracerProvider` + `OTLPSpanExporter` + LlamaIndex dispatcher span handler.
- When disabled: no tracing imports configure anything (zero overhead).

### `app/routers/upload.py`, `app/routers/chat.py`

- Call sites renamed to `build_rag_index` / `query_rag`; `SessionNotIndexedError` import path unchanged. Schemas, history router, db_service, frontend: untouched.

## Storage

- Single pgvector table `rag_nodes` in the existing `doc_chat` database (same server as app tables; LlamaIndex manages the table schema).
- Session isolation by metadata filter (`session_id` ExactMatchFilter) — no per-session tables.
- HNSW index created by `PGVectorStore.from_params(..., hnsw_kwargs={"hnsw_m", "hnsw_ef_construction", "hnsw_ef_search", "hnsw_dist_method": "vector_cosine_ops"})`, values from Settings.
- `CREATE EXTENSION IF NOT EXISTS vector;` run once (compose `teja` user is a superuser). PGVectorStore also attempts this on initialize.
- Restart survival: indexes load lazily from pgvector on demand (same behavior as Chroma today). In-memory cache rebuilt on first query.
- Legacy `chroma_db/` directory: dead weight, left in place (user deletes later if desired).

## Dependencies (`backend/requirements.txt`)

- Remove: `langchain`, `langchain-chroma`, `langchain-community`, `langchain-google-genai`
- Add: `llama-index-core`, `llama-index-llms-gemini`, `llama-index-embeddings-fastembed`, `llama-index-vector-stores-postgres`, `traceloop-sdk`
- Keep: `fastembed`, `pypdf` (now used directly), `psycopg2-binary` (PGVectorStore sync driver), `asyncpg` (app data), all web/db/test deps
- Pin top-level LlamaIndex packages; let pip resolve a consistent `llama-index-core`

## Infra & env

- `docker-compose.yml`: postgres `image: pgvector/pgvector:pg16` (drop-in; existing `postgres_data` volume preserved).
- `backend/.env.example` additions (all optional):

```
LANGSMITH_API_KEY=
LANGSMITH_TRACING=false
LANGSMITH_PROJECT=rag-doc-chat
SENTENCES_PER_CHUNK=6
SENTENCE_OVERLAP=1
HNSW_M=16
HNSW_EF_CONSTRUCTION=64
HNSW_EF_SEARCH=40
```

(`backend/.env`: user adds a real `LANGSMITH_API_KEY` when enabling tracing.)

## Tests (offline — no DB writes, no model downloads, no API calls)

- `test_node_parser.py` (replaces `test_document_processor.py`): sentence-group chunking, section metadata, `session_id` injection, `doc > section` prefix, heading heuristics regression cases.
- `test_rag_service.py`: never-indexed path — retriever mocked to return zero nodes → `SessionNotIndexedError`; no Chroma import remains.
- `test_llm_factory.py`: updated for `Gemini` return type; registry assertions largely unchanged.
- `test_chat.py`, `test_upload.py`, `test_history.py`: import-path updates only.

## Notebook

`backend/notebooks/rag_pipeline_experiments.ipynb` rebuilt stage-per-cell in LlamaIndex idiom:

0. Setup: path setup, `load_dotenv`, verify keys
1. LOAD: `pypdf` pages, show count + preview
2. PARSE: `SectionNodeParser` — sections, sentence grouping, node metadata
3. EMBED: `FastEmbedEmbedding`, show 384-dim vector
4. STORE: `PGVectorStore` + HNSW kwargs → `VectorStoreIndex`
5. RETRIEVE: retriever with session filter, show scores + section/page metadata
6. ANSWER: query engine (`Gemini`), ask about the resume
7. PLAYGROUND: tweak `sentences_per_chunk`, `similarity_top_k`, `hnsw_ef_search`

## Tradeoffs deliverable

`docs/llamaindex-vs-langchain-tradeoffs.md`: abstractions (LCEL chains vs query engines/node parsers), observability paths (native LangSmith vs OTLP detour), integration ecosystems, code-size delta, where each framework is stronger. Written from what the migration actually surfaced.

## Verification

1. `pytest` green
2. `docker compose up postgres` (pgvector image) → `CREATE EXTENSION IF NOT EXISTS vector;`
3. Upload PDF → ask → answer cites `p.N · Section` (frontend unchanged)
4. Restart backend → same session answers (pgvector lazy reload)
5. LangSmith project shows retrieval + Gemini spans for a chat call
6. Change `HNW_EF_SEARCH` env → re-ask → observable difference in retrieval scores in the trace

## Risks / mitigations

- `traceloop-sdk` LlamaIndex instrumentation gaps → fallback to manual OTel provider + LlamaIndex dispatcher span handler (both documented paths)
- Regex sentence splitter edge cases (abbreviations, bullet lists in resumes) → bullets rarely end with `.`; oversized/undersized chunks only degrade retrieval granularity, never break it; regressions caught by `test_node_parser.py`
- pgvector HNSW on tiny per-session datasets behaves differently than at scale → fine for learning; ef_search knobs exist precisely to explore this
- `llama-index-llms-gemini` API differences vs `langchain-google-genai` (e.g., response object shape) → contained in `rag_service.py` + `llm_factory.py`

## Out of scope

- LangSmith evaluation datasets / experiments
- LlamaIndex agents, tools, routers
- Deleting `chroma_db/` or postgres volume migration
- Frontend changes
- Dockerizing backend/frontend (per no-Docker-until-upgrade-done constraint; only the postgres service runs in compose)
