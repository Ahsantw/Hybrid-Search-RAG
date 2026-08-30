"""
Runs the eval_cases.json test set against two retrieval strategies built on
the SAME LLM and the SAME FAISS index (so the only variable under test is
retrieval quality):

  - baseline: plain dense (FAISS embedding) retrieval, top-k. This reproduces
    the exact retrieval + prompting behaviour of baseline_code/rag_cli.py
    (same "stuff" chain, same prompt template, same k). We don't literally
    execute baseline_code/rag_cli.py because (a) it's an interactive input()
    loop, not a batch-callable function, and (b) its unpinned
    `from langchain.chains import RetrievalQA` style imports are from the
    pre-split langchain layout and no longer resolve against the
    langchain 1.x / langchain_classic packages installed in this
    environment. src/vector_db.py's PDFVectorStore.read_db() + as_retriever()
    is unchanged from baseline_code, so this is a faithful reproduction, not
    an approximation.

  - agent: the current repo's hybrid retriever (PDFVectorStore.get_hybrid_retriever,
    BM25 + dense via reciprocal rank fusion) -- the actual improvement made
    on top of the baseline.

Both variants load the LLM and FAISS index once and share them, so only the
retriever differs between the two RetrievalQA chains.

Scoring is deterministic (keyword / source-file substring matching), not an
LLM-judge, since no external grading model or API key is configured in this
local OpenVINO/CPU setup. See score_case() for the exact rules. This keeps
the eval reproducible and free to re-run, at the cost of being a proxy for
true answer quality rather than a full semantic judgment -- treat scores as
directional signal, and read the raw answers in the results JSON for the
real comparison.

Usage (from the project root, in the `rag` environment):
    python eval/run_eval.py
"""

import argparse
import json
import os
import sys
import time

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import yaml
from langchain_classic.chains import RetrievalQA
from langchain_classic.globals import set_verbose
from langchain_classic.prompts import PromptTemplate

from src.convert_llama_to_open import OpenVINOLLMLoader
from src.log_setup import setup_logger
from src.vector_db import PDFVectorStore

set_verbose(False)
logger = setup_logger("eval", "")

PROMPT_TEMPLATE = PromptTemplate.from_template(
    """<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are an expert assistant. Use the following extracted parts of a document to answer the question accurately and concisely.
    If the answer is not found, say you don't know. Don't try to make up an answer.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Context:
    {context}

    Question: {question}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
)

REFUSAL_PHRASES = [
    "don't know",
    "do not know",
    "no information",
    "not contain",
    "does not contain",
    "cannot find",
    "can't find",
    "not mentioned",
    "not found in",
    "unable to find",
    "no mention",
    "not provided in",
    "not aware of",
]


def build_chain(llm_model, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT_TEMPLATE},
        return_source_documents=True,
        verbose=False,
    )


def run_case(chain, case):
    start = time.time()
    result = chain.invoke(case["question"])
    elapsed = time.time() - start

    answer = result["result"]
    sources = [
        {
            "file": str(doc.metadata.get("source", "unknown")),
            "page": str(doc.metadata.get("page", "N/A")),
        }
        for doc in result["source_documents"]
    ]
    return answer, sources, elapsed


def score_case(case, answer, sources):
    answer_lower = answer.lower()
    source_files = [s["file"] for s in sources]

    if case["type"] == "refusal":
        hit = any(phrase in answer_lower for phrase in REFUSAL_PHRASES)
        return {
            "refusal_score": 1.0 if hit else 0.0,
            "overall_score": 1.0 if hit else 0.0,
        }

    keywords = case["expected_keywords"]
    keyword_hits = [kw for kw in keywords if kw.lower() in answer_lower] if keywords else []
    keyword_score = (len(keyword_hits) / len(keywords)) if keywords else None

    expected_sources = case["expected_sources"]
    source_hits = [
        exp for exp in expected_sources if any(exp in f for f in source_files)
    ] if expected_sources else []
    source_score = (len(source_hits) / len(expected_sources)) if expected_sources else None

    parts = [s for s in (keyword_score, source_score) if s is not None]
    overall = sum(parts) / len(parts) if parts else 0.0

    return {
        "keyword_score": keyword_score,
        "keyword_hits": keyword_hits,
        "source_score": source_score,
        "source_hits": source_hits,
        "overall_score": overall,
    }


def run_variant(name, chain, cases):
    print(f"\n=== Running {name} ===")
    results = []
    for case in cases:
        print(f"  [{name}] {case['id']}: {case['question'][:70]}...")
        answer, sources, elapsed = run_case(chain, case)
        scores = score_case(case, answer, sources)
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "question": case["question"],
                "answer": answer.strip(),
                "sources": sources,
                "response_time_seconds": elapsed,
                "scores": scores,
            }
        )
        print(f"    -> {elapsed:.1f}s, overall_score={scores['overall_score']:.2f}")
    return results


def aggregate(results):
    n = len(results)
    avg_overall = sum(r["scores"]["overall_score"] for r in results) / n
    avg_latency = sum(r["response_time_seconds"] for r in results) / n

    keyword_scores = [r["scores"]["keyword_score"] for r in results if r["scores"].get("keyword_score") is not None]
    source_scores = [r["scores"]["source_score"] for r in results if r["scores"].get("source_score") is not None]
    refusal_scores = [r["scores"]["refusal_score"] for r in results if r["scores"].get("refusal_score") is not None]

    return {
        "avg_overall_score": avg_overall,
        "avg_response_time_seconds": avg_latency,
        "avg_keyword_score": (sum(keyword_scores) / len(keyword_scores)) if keyword_scores else None,
        "avg_source_score": (sum(source_scores) / len(source_scores)) if source_scores else None,
        "avg_refusal_score": (sum(refusal_scores) / len(refusal_scores)) if refusal_scores else None,
        "num_cases": n,
    }


def write_comparison_table(cases, baseline_results, agent_results, baseline_agg, agent_agg, out_path):
    by_id_baseline = {r["id"]: r for r in baseline_results}
    by_id_agent = {r["id"]: r for r in agent_results}

    lines = []
    lines.append("# Baseline vs Agent — Comparison Table\n")
    lines.append(
        "Baseline = plain dense (FAISS) retrieval, reproducing `baseline_code/rag_cli.py`'s "
        "retrieval + prompting exactly. Agent = current repo's hybrid BM25 + dense retrieval "
        "(`src/vector_db.py: PDFVectorStore.get_hybrid_retriever`). Same LLM, same FAISS index, "
        "same prompt template, same k in both — retrieval strategy is the only variable.\n"
    )
    lines.append(
        "Scores are deterministic keyword/source-match checks (0-1), not an LLM judge — see "
        "`run_eval.py` docstring. Treat as directional signal; the raw answers in "
        "`results/baseline_results.json` / `results/agent_results.json` are the ground truth "
        "for actually judging quality.\n"
    )

    lines.append("## Summary\n")
    lines.append("| Metric | Baseline (dense-only) | Agent (hybrid BM25+dense) |")
    lines.append("|---|---|---|")
    lines.append(f"| Avg overall score | {baseline_agg['avg_overall_score']:.2f} | {agent_agg['avg_overall_score']:.2f} |")
    if baseline_agg["avg_keyword_score"] is not None:
        lines.append(f"| Avg keyword recall | {baseline_agg['avg_keyword_score']:.2f} | {agent_agg['avg_keyword_score']:.2f} |")
    if baseline_agg["avg_source_score"] is not None:
        lines.append(f"| Avg source-hit rate | {baseline_agg['avg_source_score']:.2f} | {agent_agg['avg_source_score']:.2f} |")
    if baseline_agg["avg_refusal_score"] is not None:
        lines.append(f"| Refusal accuracy (out-of-scope case) | {baseline_agg['avg_refusal_score']:.2f} | {agent_agg['avg_refusal_score']:.2f} |")
    lines.append(f"| Avg response time (s) | {baseline_agg['avg_response_time_seconds']:.1f} | {agent_agg['avg_response_time_seconds']:.1f} |")
    lines.append(f"| Cases run | {baseline_agg['num_cases']} | {agent_agg['num_cases']} |")
    lines.append("")

    lines.append("## Per-case results\n")
    lines.append("| ID | Difficulty | Question | Baseline score | Agent score | Baseline latency (s) | Agent latency (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for case in cases:
        cid = case["id"]
        b = by_id_baseline[cid]
        a = by_id_agent[cid]
        q_short = case["question"] if len(case["question"]) <= 90 else case["question"][:87] + "..."
        lines.append(
            f"| {cid} | {case['difficulty']} | {q_short} "
            f"| {b['scores']['overall_score']:.2f} | {a['scores']['overall_score']:.2f} "
            f"| {b['response_time_seconds']:.1f} | {a['response_time_seconds']:.1f} |"
        )
    lines.append("")

    hard_case = by_id_agent.get("q12")
    hard_case_baseline = by_id_baseline.get("q12")
    if hard_case and hard_case_baseline:
        lines.append("## Hard case spotlight: q12 (duplicate claim number)\n")
        lines.append(
            "`TCD 001/2024` has two separate cost orders (AED 60,000 and AED 44,000) issued on "
            "different dates. This case checks whether each system's retrieval step actually pulls "
            "chunks from *both* same-named documents, and whether generation distinguishes them "
            "instead of confidently reporting a single wrong figure.\n"
        )
        lines.append(f"- Baseline sources returned: {hard_case_baseline['sources']}")
        lines.append(f"- Baseline answer: {hard_case_baseline['answer']}")
        lines.append(f"- Agent sources returned: {hard_case['sources']}")
        lines.append(f"- Agent answer: {hard_case['answer']}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load_existing(results_dir, name):
    path = os.path.join(results_dir, f"{name}_results.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"--only skipped {name}, but {path} doesn't exist yet. Run without --only first."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"], data["aggregate"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["baseline", "agent"],
        help="Only (re)run this variant; reuse the other's existing results/*.json untouched.",
    )
    args = parser.parse_args()

    with open("config/config.yaml", "r") as f:
        full_config = yaml.safe_load(f)
    config = full_config["vector_db"]

    with open(os.path.join(EVAL_DIR, "eval_cases.json"), "r", encoding="utf-8") as f:
        cases = json.load(f)

    results_dir = os.path.join(EVAL_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    run_baseline = args.only in (None, "baseline")
    run_agent = args.only in (None, "agent")

    llm_model = None
    db_class = None
    db = None
    if run_baseline or run_agent:
        print("Loading LLM...")
        loader = OpenVINOLLMLoader(logger)
        llm_model = loader.load_openvino_llm()

        print("Loading FAISS index...")
        db_class = PDFVectorStore(logger)
        db = db_class.read_db()

    if run_baseline:
        baseline_retriever = db.as_retriever(search_kwargs={"k": config["k"]})
        baseline_chain = build_chain(llm_model, baseline_retriever)
        baseline_results = run_variant("baseline", baseline_chain, cases)
        baseline_agg = aggregate(baseline_results)
        with open(os.path.join(results_dir, "baseline_results.json"), "w", encoding="utf-8") as f:
            json.dump({"aggregate": baseline_agg, "cases": baseline_results}, f, indent=2)
    else:
        print("Skipping baseline -- reusing existing eval/results/baseline_results.json")
        baseline_results, baseline_agg = load_existing(results_dir, "baseline")

    if run_agent:
        agent_retriever = db_class.get_hybrid_retriever(
            db,
            k=config["k"],
            bm25_weight=config["bm25_weight"],
            dense_weight=config["dense_weight"],
            fetch_k=config["hybrid_fetch_k"],
        )
        agent_chain = build_chain(llm_model, agent_retriever)
        agent_results = run_variant("agent", agent_chain, cases)
        agent_agg = aggregate(agent_results)
        with open(os.path.join(results_dir, "agent_results.json"), "w", encoding="utf-8") as f:
            json.dump({"aggregate": agent_agg, "cases": agent_results}, f, indent=2)
    else:
        print("Skipping agent -- reusing existing eval/results/agent_results.json")
        agent_results, agent_agg = load_existing(results_dir, "agent")

    write_comparison_table(
        cases,
        baseline_results,
        agent_results,
        baseline_agg,
        agent_agg,
        os.path.join(results_dir, "comparison_table.md"),
    )

    print("\n=== Done ===")
    print("Baseline avg overall score:", round(baseline_agg["avg_overall_score"], 3))
    print("Agent avg overall score:   ", round(agent_agg["avg_overall_score"], 3))
    print("Results written to eval/results/")


if __name__ == "__main__":
    main()
