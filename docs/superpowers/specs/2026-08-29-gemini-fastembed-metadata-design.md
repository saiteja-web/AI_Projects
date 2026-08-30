# Gemini-only RAG upgrade with fastembed embeddings and chunk metadata

Date: 2026-08-29
Status: Approved (pending implementation plan)

## Goal

Make the project lighter and simpler by:

1. Replacing Ollama + OpenAI with Gemini as the single LLM provider
2. Replacing torch-based sentence-transformers embeddings with fastembed (ONNX, no torch)
3. Enriching chunks with section metadata to improve retrieval quality and citations
4. Keeping the existing docker-compose deployment model (postgres + backend + frontend; ollama service removed)

Motivating constraints: user's disk has ~20 GB free (Docker.raw alone is 26 GB, mostly ollama models and torch-laden backend image); user wants to keep containerized deployment for now; notebook `backend/notebooks/rag_pipeline_experiments.ipynb` is for learning pipeline stages.

## Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| LLM provider | Gemini only (remove Ollama + OpenAI everywhere) |
| Gemini lineup | `gemini-2.5-flash` (default), `gemini-2.5-pro`, `gemini-2.5-flash-lite` |
| API shape | Collapse to model-only: no `provider` field in API, schemas, cache keys |
| Embeddings | fastembed local, model `BAAI/bge-small-en-v1.5` (384-dim, ONNX, no torch) |
| Chunk metadata | Section headers detected at load time; metadata fields (section, page, source, chunk_index) + section path prepended to embedded text |
| Deployment | Keep docker-compose (postgres, backend, frontend); delete ollama service |
| API key | `GEMINI_API_KEY` in `backend/.env` (user provides) |

## Backend changes

### `app/services/llm_factory.py`
- `ModelInfo`: drop `provider` field; keep id, name, context_tokens, description
- `AVAILABLE_MODELS`: the 3 Gemini models above
- `create_llm(model_id, temperature)` → `ChatGoogleGenerativeAI(model=model_id, google_api_key=settings.gemini_api_key, temperature=temperature)`
- `get_default_model()` → returns `"gemini-2.5-flash"` (string, not tuple)
- Delete `get_models_for_provider`

### `app/core/config.py`
- Remove: `ollama_base_url`, `openai_api_key`, `default_llm_provider`
- Add: `gemini_api_key: str = ""`
- Change: `default_llm_model` → `"gemini-2.5-flash"`; `embedding_model` → `"BAAI/bge-small-en-v1.5"`

### `app/services/rag_service.py`
- `CHROMA_DIR` default: `./chroma_db` — resolves correctly in both contexts (host-native runs from `backend/`; in the container cwd is `/app`, which is the mounted `chroma_data` volume), so no env override is needed
- `_embeddings()` → `FastEmbedEmbeddings(model=settings.embedding_model)` from `langchain_fastembed`
- Chain cache: `rag_chains[session_id][model_id]`
- `build_rag_chain(file_path, session_id, model_id=...)`, `query_rag_chain(session_id, query, model_id=None)`
- **Indexing pipeline (new shape):**
  1. `PyPDFLoader.load()` → pages
  2. Section detection: walk lines of each page; a heading is a line that is short (<60 chars), has no terminal period, and is title/UPPER case; carry the current section forward across pages
  3. Build one Document per section with metadata `{source, page, section}` (concatenating that section's text on the page), then `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` per section-document (splitter propagates parent metadata to child chunks)
  4. Post-split: assign sequential `chunk_index`; set `page_content = f"{document_name} > {section}\n\n{chunk_text}"` so embeddings AND the LLM context carry section context
- `sources` returned by `query_rag_chain` becomes `list[{"page": int, "section": str}]` (deduped)
- Drop `_llm` provider arg; `format_docs` unchanged

### `app/models/schemas.py`
- `ChatRequest`: remove `provider`; keep `model_id: str | None = None`
- `ChatResponse.sources`: `list[dict]` (page + section)

### `app/routers/models.py`
- Return `{"models": [ {id, name, context_tokens, description}, ... ], "default_model": "gemini-2.5-flash"}`

### `app/routers/chat.py`, `app/routers/upload.py`
- Update call sites for new signatures (no provider arg)
- `db_service.save_message` stores sources as JSON — new dict shape needs no schema change

## Dependencies

`backend/requirements.txt`:
- Remove: `langchain-ollama`, `langchain-openai`, `langchain-huggingface`, `sentence-transformers`
- Add: `langchain-google-genai`, `langchain-fastembed` (pin versions pip resolves; keep the langchain family on a compatible core)
- Keep: `pypdf`, everything else

## Docker / env

- `backend/Dockerfile`: delete the CPU-torch install step (lines 10–14) — no longer needed
- `docker-compose.yml`: delete `ollama` service; backend no longer sets `OLLAMA_BASE_URL` or depends on ollama; `GEMINI_API_KEY` flows via `env_file` (`backend/.env`)
- `backend/.env.example`: `DATABASE_URL` (unchanged, compose hostname), `GEMINI_API_KEY=`, `DEFAULT_LLM_MODEL=gemini-2.5-flash`, `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`, remove Ollama/OpenAI/provider lines
- `backend/.env`: user pastes their real `GEMINI_API_KEY`; Ollama/OpenAI lines removed
- Old chroma vectors are incompatible (different embedding model) → start with a fresh chroma volume (`docker compose down -v` before first `up`)

## Frontend (small)

- `src/types/index.ts`: models response type = flat list; drop provider types
- `src/services/api.ts`: chat request body sends `model_id` only
- `src/components/ChatPage.tsx`: single-level model selector (3 Gemini models); sources render as `p.{page} · {section}`

## Notebook `backend/notebooks/rag_pipeline_experiments.ipynb`

One stage per cell with printed intermediates (learning goal):
0. Setup: import path to `backend/`, `load_dotenv("../.env")`, verify `GEMINI_API_KEY`
1. LOAD: `PyPDFLoader` on the resume (quoted path), show page count + text preview
2. CHUNK: section detection + 500/50 split; show chunk count, metadata, sample chunks with section prefixes
3. EMBED: `FastEmbedEmbeddings`, embed a sample chunk, show vector dims
4. STORE: `Chroma.from_documents` into `rag_experiment` collection (local `./chroma_db`)
5. RETRIEVE: `similarity_search` with scores; show each hit's section/page metadata
6. ANSWER: LCEL chain (retriever → prompt → `ChatGoogleGenerativeAI(gemini-2.5-flash)` → parser); ask a question about the resume
7. Playground: tweak `k`, chunk size, and watch retrieval change

## Verification

1. `pytest` (update tests referencing ollama/openai/provider; run the suite the same way it runs today)
2. `docker compose up --build` → `GET /models` shows 3 Gemini models
3. Upload resume PDF via UI → ask a question → answer cites `p.N · Section`
4. Notebook runs end-to-end with the resume PDF

## Risks / mitigations

- `langchain-fastembed` compatibility with the pinned langchain 1.x family — pin whatever resolves; fallback (if broken): use `fastembed` directly behind a ~30-line LangChain `Embeddings` wrapper
- Heading heuristics are tuned for prose-style PDFs (resume-like); imperfect detection only degrades metadata quality, never breaks retrieval
- Gemini API region/availability — standard Google API, no special setup beyond the key

## Out of scope

- Metadata filter parameters on the chat API (future)
- Pinecone or any other vector store
- Deleting old Docker images/volumes (user-initiated cleanup after rebuild; Docker Desktop "Clean / Purge data" reclaims most of the 26 GB)
