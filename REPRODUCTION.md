# Reproduction Guide

Step-by-step instructions to set this repo up from scratch on a new machine: CLI pipeline, web UI (backend + frontend), and the eval harness. Follow in order — each stage depends on the one before it.

## 1. What you're setting up

A local RAG pipeline over your own PDFs:
- **LLM**: Llama-3.1-8B-Instruct, converted once to OpenVINO INT4 and then loaded from disk on every run (no re-download/re-convert needed after the first time).
- **Retrieval**: FAISS (dense embeddings) + BM25 (keyword), combined via reciprocal rank fusion.
- **Interfaces**: a terminal CLI (`rag_cli.py`), a FastAPI + React web UI (`backend/`, `frontend/`) with PIN auth and streaming answers, and an eval harness (`eval/`) that scores retrieval quality.

Everything runs on CPU — despite the hardware specs table further down mentioning a GPU, the LLM pipeline is hardcoded to `device=-1` (CPU) in `src/convert_llama_to_open.py`, so no GPU is required or used.

## 2. Prerequisites

- **Python 3.12** (this repo was built and verified against 3.12.14; the older README note about Python 3.10 is not what was actually tested here — 3.10/3.11 will likely work too but aren't verified).
- **~6 GB free disk** — the converted OpenVINO model alone is ~4.4 GB, plus PDFs and the FAISS index.
- **~8 GB+ RAM** free while running — the 8B-parameter model is loaded fully into memory even quantized to INT4.
- **Node.js + npm** (only needed for the web frontend). Verified against Node 24 / npm 11; anything reasonably recent should work.
- **A Hugging Face account with access to `meta-llama/Llama-3.1-8B-Instruct`** (only needed the *first* time you convert the model — see step 5). If you already have a pre-converted `models/llama-3.1-instruct-8b-ovir-int4/` folder, you can skip this entirely.
- **conda** (recommended) or plain `venv` — this guide uses conda since that's what the working dev environment on this machine uses, but a plain `python -m venv` + `pip install` works identically.

## 3. Get the code and the data

Copy/clone the repo, then decide what documents to index:

- The repo currently ships with 12 real DIFC Courts case PDFs in `data/` (used to build the eval set in `eval/eval_cases.json`) plus an already-built `db/faiss_index/` matching them. If you keep these as-is, you can skip straight to step 6.
- To index your **own** documents instead: delete or replace the PDFs in `data/`, then rebuild the FAISS index in step 5 (`python src/vector_db.py`). Note that if you swap documents, `eval/eval_cases.json` no longer matches reality — the eval harness assumes the current DIFC Courts case set (see step 8).

## 4. Create the Python environment and install dependencies

```bash
conda create -n rag python=3.12 -y
conda activate rag
pip install -r requirements.txt
```

`requirements.txt` pins the heavy/version-sensitive packages (`openvino`, `optimum`, `optimum-intel`, `nncf`, `torch`, `transformers`) but leaves the LangChain family unpinned. LangChain has gone through a breaking package split (the old `langchain.chains`/`langchain.prompts` import style used by `baseline_code/` no longer resolves against current releases — this repo's own code uses `langchain_classic.chains`/`langchain_classic.prompts` instead specifically because of that split). If a fresh install ever breaks on an import, the exact versions verified working together on this machine are:

```
langchain==1.3.18
langchain-classic==1.0.8
langchain-community==0.4.2
langchain-core==1.6.1
langchain-huggingface==1.2.2
langchain-openai==1.6.0
langchain-text-splitters==1.1.2
faiss-cpu==1.15.0
fastapi==0.141.1
uvicorn==0.52.4
sentence-transformers==5.7.0
pypdf==6.16.2
rank-bm25==0.2.2
openvino==2025.0.0
optimum==1.24.0
optimum-intel==1.22.0
nncf==2.15.0
torch==2.6.0
transformers==4.48.3
```

If you need to pin, `pip install "langchain==1.3.18" "langchain-classic==1.0.8" ...` (etc.) before or after `pip install -r requirements.txt`.

## 5. Get the model and build the index

**Skip this step entirely if `models/llama-3.1-instruct-8b-ovir-int4/` and `db/faiss_index/` already exist** (both ship in this repo already built) — go to step 6.

Otherwise, from the project root:

```bash
# Only needed if models/llama-3.1-instruct-8b-ovir-int4/ doesn't already exist:
huggingface-cli login   # paste an HF token from https://huggingface.co/docs/hub/en/security-tokens
# then request access at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (a few minutes to be granted)

python src/convert_llama_to_open.py   # downloads + quantizes to OpenVINO INT4, saves to models/
python src/vector_db.py               # chunks the PDFs in data/ and builds db/faiss_index/
```

`src/convert_llama_to_open.py` automatically falls back to downloading + converting from Hugging Face only if it can't load an existing model from `models/llama-3.1-instruct-8b-ovir-int4/` — so re-running it is always safe and won't re-download if the folder is already there. Alternatively, `bash run_demo.sh` runs both of these steps plus the CLI in one shot.

## 6. Configure

Open `config/config.yaml` before running anything for real:

- **`auth.pin`** — defaults to `"1234"`. Change this before letting anyone besides you use the web UI; it gates the chat endpoints.
- **`vector_db.k`** — how many chunks are retrieved per question (default `3`).
- **`vector_db.bm25_weight` / `dense_weight` / `hybrid_fetch_k`** — hybrid retrieval tuning (defaults `0.4` / `0.6` / `8`); see `eval/results/comparison_table.md` for how these were arrived at if you plan to change them.
- **`llm.max_new_tokens` / `temperature` / `top_p`** — generation behavior.

## 7. Run it

**Terminal CLI:**
```bash
python rag_cli.py
```

**Web UI** (two terminals, both from the project root, same Python environment for the backend):
```bash
# Terminal 1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (default `http://127.0.0.1:5173`), enter the PIN from `config/config.yaml`, and start asking questions. The dev server proxies `/api/*` to the backend on port 8000 — `frontend/vite.config.js` binds explicitly to `127.0.0.1` (Vite's IPv6-loopback default caused connection issues during development; this is already fixed in the shipped config, just noting it in case you ever reset that file).

Expect **~15-30 seconds per answer** on CPU — this is inherent to running an 8B model without a GPU, not a bug.

## 8. Run the eval harness (optional)

```bash
python eval/run_eval.py
```

This loads the LLM once, then runs all 13 cases in `eval/eval_cases.json` through both a baseline (plain dense FAISS) retriever and the current hybrid retriever, and writes `eval/results/{baseline,agent}_results.json` + `comparison_table.md`, plus a per-question `retrieval → verification → answer` trace for each variant to `trajectories/eval/{baseline,agent}_trajectory.jsonl` (one JSON line per question, overwritten each run — useful for inspecting exactly which chunks were retrieved and why a verification verdict landed where it did). It takes roughly **10-15 minutes** (26 LLM generations at ~20s each). To re-run just one side without touching the other's saved results:

```bash
python eval/run_eval.py --only agent
python eval/run_eval.py --only baseline
```

The eval questions are grounded in the specific facts of the 12 DIFC Courts PDFs shipped in `data/` — if you replace those documents, the eval cases will no longer make sense and would need to be rewritten against your own corpus.

## 9. Common issues

**"You are trying to access a gated repo"** during model conversion — you haven't been granted access to `meta-llama/Llama-3.1-8B-Instruct` yet. Request it at the model's Hugging Face page, wait a few minutes for approval, then re-run `huggingface-cli login` and `python src/convert_llama_to_open.py`.

**`ModuleNotFoundError: No module named 'langchain.chains'`** — you're on a LangChain version from after the package split trying to use the old-style import path. This repo's own code already uses the correct `langchain_classic.*` imports; this error only shows up if you're trying to run `baseline_code/rag_cli.py` directly (it uses the old import style intentionally, as a frozen historical baseline — it isn't meant to run standalone against a modern LangChain install; see `eval/run_eval.py`'s docstring for why the eval harness reproduces its behavior instead of executing it directly).

**Frontend loads but can't reach the backend / requests hang** — confirm the backend is actually running and healthy first: `curl http://127.0.0.1:8000/api/health` should return `{"status":"ok","ready":true}`. If `ready` is `false`, the LLM/index are still loading — wait and retry.

**Locked out of the PIN login** — 5 wrong attempts locks that IP out for 5 minutes (see `backend/main.py`). Wait it out, or restart the backend to clear the in-memory lockout state during local testing.

**Port 8000 or 5173 already in use** — another instance is probably already running (check `Get-NetTCPConnection -State Listen` on Windows, `lsof -i :8000` on Linux/macOS) — stop it, or pass a different `--port` to uvicorn / `--port` to `vite`.
