# LlamaIndex vs LangChain — observed tradeoffs

Notes from converting this repo's RAG pipeline (PDF → section/sentence nodes →
pgvector HNSW → Gemini) from LangChain 1.x to LlamaIndex, August 2026.

## What changed concretely

| Concern | LangChain (before) | LlamaIndex (after) |
|---|---|---|
| Orchestration | LCEL pipe: `retriever \| format \| prompt \| llm \| parser` | retriever + response synthesizer composed by hand |
| Document model | `Document{page_content, metadata}` | `TextNode{text, metadata}` (+ node relationships) |
| Custom chunking | compose `PyPDFLoader` + `RecursiveCharacterTextSplitter` | subclass `NodeParser` (first-class hook, `_parse_nodes`) |
| Vector store | `langchain-chroma` (local files) | `PGVectorStore` (HNSW in the existing Postgres) |
| LLM | `ChatGoogleGenerativeAI` | `GoogleGenAI` (same 3-model registry) |
| Prompt | f-string template | `PromptTemplate` with `{context_str}`/`{query_str}` conventions |
| Answer flow | `chain.invoke(query)` + separate `retriever.invoke(query)` (retrieval ran twice) | `retriever.retrieve(query)` then `synthesize(query, nodes=...)` (retrieval once; citations = exactly the used nodes) |
| Observability | LangSmith native callbacks | OTLP via OpenLLMetry → LangSmith OTel endpoint |

Code size: the backend RAG pipeline churned 638 lines across 11 files
(`git diff --stat 62b6cc3..HEAD -- backend/app`).

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
   OTLP endpoint. Works, but it is one more moving part — and the integration
   SDK (traceloop) needed explicit `api_endpoint`/`headers` params because it
   ignores the standard `OTEL_EXPORTER_OTLP_*` env vars.
4. **Citations got simpler.** Retrieved nodes carry scores + metadata, so
   `p.N · section` citations come from the exact nodes the answer used — no
   second retrieval pass like the LCEL version needed.
5. **Vector store choice is where the leverage is.** Moving Chroma→pgvector
   bought HNSW tunables (`m`, `ef_construction`, `ef_search` — all env-tunable
   here) and dropped a storage engine; the knobs live in config, not code.
6. **Ecosystem maturity has edges.** Two concrete snags: the
   `llama-index-embeddings-fastembed` integration requires Python <3.13 and
   pins `fastembed<0.2`, so we wrote a ~30-line custom `BaseEmbedding` wrapper
   (which is also a nice lesson in how integrations work); and the Gemini LLM
   constructor makes a live API metadata call unless `context_window` and
   `max_tokens` are passed explicitly. Reading the installed integration source
   was necessary — docs lag the code.
7. **Ecosystem fit.** LangChain feels broader (agents, tools, 100+ loaders);
   LlamaIndex feels deeper on the ingest→index→retrieve path. For a RAG app
   like this one, the depth is the better fit.

## When to pick which

- LlamaIndex: document-centric apps, custom ingest/chunking, retrieval quality
  tuning, vector-store experimentation.
- LangChain: agent/tool orchestration, when the team already speaks LCEL, or
  when LangSmith-native tracing matters more than retrieval depth.
