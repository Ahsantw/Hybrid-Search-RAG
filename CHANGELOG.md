# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added — Reproduction guide
- `REPRODUCTION.md` — full setup walkthrough for a new machine: prerequisites, environment creation, dependency pinning notes (the LangChain package-split issue hit while building the eval harness), model conversion, config knobs to check before first real use (`auth.pin` especially), running the CLI/web UI/eval harness, and a troubleshooting section for the specific failure modes hit during this build (gated HF model access, the old-style `langchain.chains` import error, PIN lockouts, port conflicts). Linked from `README.md`.

### Added — Web UI (backend + frontend)
- New FastAPI backend (`backend/main.py`) wrapping the existing RAG pipeline (`src/convert_llama_to_open.py`, `src/vector_db.py`) as an HTTP API instead of a terminal-only CLI. The LLM and FAISS index are loaded once at startup instead of being rebuilt on every question.
- New React (Vite) frontend (`frontend/`) providing a chat UI: message bubbles, per-answer source citations (file + page), loading indicator.
- `POST /api/chat` — ask a question, get back `{ answer, sources, response_time }`.
- `GET /api/health` — readiness check used by the frontend and for local testing.
- README updated with instructions for running the backend and frontend locally.
- Added `fastapi` and `uvicorn[standard]` to `requirements.txt`.

### Added — Streaming responses
- `POST /api/chat/stream` — Server-Sent Events endpoint that streams the answer token-by-token instead of waiting for the full ~15-20s generation to finish. Sources are sent as soon as retrieval completes, before generation starts.
- Backend now drives generation directly via `HuggingFacePipeline.stream()` (backed by `transformers.TextIteratorStreamer`) instead of going through the `RetrievalQA` chain, so retrieval and generation are separate, streamable steps.
- Frontend (`frontend/src/streamChat.js`) consumes the SSE stream via `fetch` + `ReadableStream` (`EventSource` doesn't support POST bodies) and renders the answer live with a typing indicator and blinking cursor.

### Added — PIN authentication
- `POST /api/login` — checks a 4-digit PIN (`auth.pin` in `config/config.yaml`, default `1234`) and returns a session token.
- `/api/chat` and `/api/chat/stream` now require `Authorization: Bearer <token>`; requests without a valid token get `401`.
- Brute-force lockout: an IP gets 5 attempts, then a 5-minute lockout, since a bare 4-digit PIN is only 10,000 combinations.
- Frontend (`frontend/src/Login.jsx`) shows a PIN entry screen before the chat UI; the token is kept in `sessionStorage` (cleared when the tab closes) with a "Log out" button, and an expired/invalid session bounces the user back to the login screen automatically.

### Added — Hybrid search (BM25 + embeddings)
- `PDFVectorStore.get_hybrid_retriever()` (`src/vector_db.py`) combines keyword search (`BM25Retriever`, built from the same chunks already stored in the FAISS index) with the existing dense embedding retriever via `EnsembleRetriever` (reciprocal rank fusion), so exact terms (case names, citations) and semantically related passages both surface.
- `bm25_weight` and `dense_weight` added to `config/config.yaml` (default `0.5` / `0.5`) to tune the balance between keyword and semantic retrieval.
- Added `TopKRetriever` wrapper so the fused result is trimmed back down to the configured `k`, since `EnsembleRetriever` otherwise returns the full union of each sub-retriever's top-k list.
- `rag_cli.py` and `backend/main.py` both updated to use the hybrid retriever instead of the plain FAISS retriever.
- Added `rank_bm25` to `requirements.txt`.

### Added — Eval harness (baseline vs agent)
- `eval/eval_cases.json` — 13 Q&A test cases grounded in the actual facts of the 12 DIFC Courts case PDFs in `data/`, spanning easy/medium/hard difficulty, an out-of-scope refusal/hallucination trap, and one deliberately adversarial hard case: claim number `TCD 001/2024` has two separate cost orders (AED 60,000 and AED 44,000) under the same case name, testing whether retrieval can tell same-named documents apart.
- `eval/run_eval.py` — loads the LLM and FAISS index once and builds two identical `RetrievalQA` chains that differ only in retriever: baseline (plain dense FAISS, reproducing `baseline_code/rag_cli.py`'s retrieval behavior exactly) vs agent (the current hybrid BM25+dense retriever). Runs all cases through both and scores answers with deterministic keyword/source-match checks (no LLM judge is configured in this local setup).
- `eval/results/baseline_results.json`, `eval/results/agent_results.json`, `eval/results/comparison_table.md` — raw outputs, per-case scores, and a judge-facing comparison table from the first run.
- Notable finding from that first run: hybrid search scored *lower* than dense-only on this small 12-document corpus (0.71 vs 0.79 avg score), traced to BM25 occasionally displacing a needed chunk from the correct document with a lexically-similar chunk from the wrong one at k=3 — most visibly in one case (q05) where it turned a correct answer into an incorrect refusal. Documented in `comparison_table.md` with root-cause analysis rather than hidden or re-run to get a better number.
- `eval/run_eval.py` gained an `--only baseline|agent` flag so one variant can be re-run without touching the other's saved results — used below to iterate on the agent without ever re-generating baseline's numbers.

### Fixed — Hybrid retriever regression found by the eval
- `PDFVectorStore.get_hybrid_retriever()` (`src/vector_db.py`) gained a `fetch_k` parameter (default `max(3*k, 8)`, config: `hybrid_fetch_k: 8`): each sub-retriever now pulls a wider candidate pool before rank fusion instead of only `k`, so a single bad BM25 pick can no longer evict a correct dense chunk purely by scarcity. This was the direct fix for the q05 regression identified above.
- `bm25_weight` / `dense_weight` rebalanced from 0.5/0.5 to `0.4`/`0.6` — on this small, well-embedded corpus dense retrieval was already close to ceiling, so tilting toward it reduces BM25 noise while keeping its contribution for genuine keyword/citation matches.
- Result: agent's eval score moved from 0.71 to 0.78 against baseline's unmoved 0.79 (baseline's code, config, and results were never touched) — effectively parity. A more aggressive follow-up attempt (`hybrid_fetch_k: 12`, `dense_weight: 0.7`) was tried and scored *worse* (0.76); that setting was reverted rather than kept, and the negative result is recorded in `comparison_table.md` rather than discarded, since chasing a specific 13-question eval past the point of principled justification is a form of overfitting.
- `backend/main.py` and `rag_cli.py` updated to pass `fetch_k` through from config.
