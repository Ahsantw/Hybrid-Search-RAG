## Problem & User

**Who has this problem?**
Lawyers and legal teams — in this case, working with DIFC Courts case documents — deal with 
large volumes of dense legal PDFs: case files, judgments, cost orders, and filings that can 
run into hundreds of pages across dozens of documents. Finding a specific fact (a claim 
number, a cost order amount, a ruling detail) means manually searching or skimming through 
these documents case by case.

**What's the bottleneck?**
Three things make this genuinely hard to solve with an off-the-shelf tool:

- **Volume** — a single matter can span many long PDFs, and the relevant fact could be 
  buried on any page of any document. Manual search doesn't scale, and it's easy to miss or 
  misattribute similar-sounding cases (e.g., two cost orders under the same case name).
- **Cost** — most AI solutions rely on paid third-party APIs charged per query. For a firm 
  running hundreds of lookups a day, that adds up fast. This solution runs entirely on local 
  CPU using an OpenVINO-optimized model, so there's no per-query API cost.
- **Privacy** — legal documents are confidential. Sending case files to a third-party API is 
  often a non-starter for client confidentiality and data-handling obligations. This solution 
  is fully in-house: the model, the retrieval index, and the documents never leave the local 
  machine.

**Why it's worth solving**
A lawyer or paralegal should be able to ask a plain-language question and get a fast, cited 
answer pulled directly from the case documents — without paying per query, without sending 
confidential filings to an external API, and without manually re-reading PDFs every time a 
question comes up.

**Does the agent solve it well?**
See [CHANGELOG.md](./CHANGELOG.md) for the full progression from a simple dense-retrieval 
baseline to the current hybrid-search + verification pipeline, and [eval/](./eval/) for 
baseline-vs-agent evidence.

**Can another person reproduce the result?**
Yes — see [REPRODUCTION.md](./REPRODUCTION.md) for full setup on a clean machine, including 
the OpenVINO model (no gated Hugging Face download required).


# Retrieval-Augmented Generation (RAG)

This project demonstrates an end-to-end Retrieval-Augmented Generation (RAG) pipeline. Main features of this repository are:

1. Download the LLaMA model (meta-llama/Llama-3.1-8B-Instruct) from Hugging Face.
2. Convert the LLaMA model to OpenVINO INT4 format.
3. If the model has already been converted, load the existing INT4 version directly from folder.
4. Use a Sentence-Transformers embedding model to generate embeddings and store them using FAISS.
5. Build a question-answering (QA) system using LangChain, with hybrid retrieval (BM25 keyword search + FAISS dense embeddings via reciprocal rank fusion) and a second-pass answer verification check (`src/verifier.py`).
6. Logs for all the steps are documented in all_logs folder. So very easy to debug any issues.
7. An eval harness (`eval/run_eval.py`) scores this hybrid+verification pipeline against a frozen dense-only baseline on 13 grounded test questions — see [REPRODUCTION.md](REPRODUCTION.md#8-run-the-eval-harness-optional) and `eval/results/comparison_table.md`.
> Setting this up on a new machine? See [REPRODUCTION.md](REPRODUCTION.md) for a complete step-by-step guide (CLI, web UI, and the eval harness).

### Installation

1. Clone the Repository
```
git clone https://github.com/Ahsantw/RAG
cd RAG
```
2. Install Python 3.12 (verified version — see [REPRODUCTION.md](REPRODUCTION.md) for details; 3.10/3.11 will likely work too but aren't verified)
3. Install required Pakages
```
pip install -r requirements.txt
```
4. Log into to hugginface accout (optional if you do not have HF model i.e. meta-llama/Llama-3.1-8B-Instruct). You will be asked to paste [HF token](https://huggingface.co/docs/hub/en/security-tokens).
```
huggingface-cli login
```
### Model

This repository uses OpenVINO INT4 version of Llama-3.1-8B-Instruct model during inference.

### Full Pipeline Inference
All steps are done with one command. At the end when all steps are done, you can easly type questions.
```
bash run_demo.sh
```
Different parameters/variable can easily be change from [config](https://github.com/Ahsantw/RAG/blob/main/config/config.yaml) file.

### Step by Step Inference
1. Download and Convert llama Model.
```
python src/convert_llama_to_open.py
```
2. Store pdf's embeding using Faiss.
```
python src/vector_db.py
```
3. Answer with reference for queries.
```
python rag_cli.py
```
Different parameters/variable can easily be change from [config](https://github.com/Ahsantw/RAG/blob/main/config/config.yaml) file.

### Sample Output
The output includes a reference from the PDF(s), followed by the answer and a verification verdict (`src/verifier.py`). This repo currently ships 12 DIFC Courts case PDFs in `data/` (see `eval/eval_cases.json` for the question set they support), so output looks like this:
```
Question ('exit'): In Oleta v Onesimo [2024] DIFC SCT 454, how much was Oleta owed for her September salary, and was Onesimo's set-off claim for training costs allowed?
 - Page 1, File: data/f4c4d051d514270964adcf58e124569e396602340b7d9172671759de10a95897.pdf
 - Page 2, File: data/f4c4d051d514270964adcf58e124569e396602340b7d9172671759de10a95897.pdf
 - Page 0, File: data/f4c4d051d514270964adcf58e124569e396602340b7d9172671759de10a95897.pdf
Answer: According to the Judgment, Oleta was owed AED 3,466.67 for her September salary, and Onesimo's set-off claim for training costs was denied.
Verification: SUPPORTED - answer is backed by the source excerpts.
Response Time: 51.6
```

### Web UI (Backend + Frontend)

The same RAG pipeline is also available as a web app so users can ask questions from a browser instead of the terminal.

- `backend/main.py` — FastAPI service that loads the LLM and FAISS index once at startup and exposes `POST /api/chat`.
- `frontend/` — React (Vite) chat UI that calls the backend.

1. Start the backend (from the project root, using the Python environment with the pipeline's dependencies installed):
```
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
2. In a separate terminal, start the frontend:
```
cd frontend
npm install
npm run dev
```
3. Open the URL Vite prints (default `http://127.0.0.1:5173`). The dev server proxies `/api/*` requests to the backend on port 8000. You'll be asked for the 4-digit PIN from `config/config.yaml` (`auth.pin`, default `1234`) before you can chat.

Each answer is followed by a self-check pass (`src/verifier.py`) that flags whether the answer is actually supported by the retrieved sources — shown as a badge under the answer. This roughly doubles response time (a second full LLM call), and it's a coarse safety net rather than a guarantee: see the "Added — Answer verification" entry in `CHANGELOG.md` for what it reliably catches and what it doesn't.

The vector DB and LLM must already be built/converted (steps above, or `bash run_demo.sh`) before starting the backend.

### Hardware Specs.

This RAG pipeline was tested successfully on the following system:

- **OS**: Windows 10/ Ubuntu 22.04 (Tested on Both)
- **Processor**: Intel Core i7 10th Gen
- **RAM**: 32 GB
- **HardDrive**: 2TB

The pipeline is CPU-only (`device=-1` is hardcoded in `src/convert_llama_to_open.py`) — no GPU is required or used, despite this machine also having one available.

### Latency
- **Answer generation only:** ~15-20 seconds (this is what the numbers above were measured on, before the verification pass existed).
- **With verification** (now always on in `rag_cli.py` and the web UI — a second full LLM call after each answer): roughly doubles to **~40-60 seconds** end-to-end. See `eval/results/comparison_table.md` for measured per-question timings (baseline avg ~21s without verification vs. agent avg ~49s with hybrid retrieval + verification).

### Common Issues
1. HugginFace login issue.
```
Failed to Run the pipeline : You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```
Solution: Go to huggingface [page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and request model's access by completing and submitting the form. It takes few minutes and they grant you access.
Then login to your hugginface accout from terminal using following command and then paste [HF token](https://huggingface.co/docs/hub/en/security-tokens).
```
huggingface-cli login
```
