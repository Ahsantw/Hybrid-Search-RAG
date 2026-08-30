# Baseline vs Agent — Comparison Table

Baseline = plain dense (FAISS) retrieval, reproducing `baseline_code/rag_cli.py`'s retrieval + prompting exactly. Agent = current repo's hybrid BM25 + dense retrieval (`src/vector_db.py: PDFVectorStore.get_hybrid_retriever`). Same LLM, same FAISS index, same prompt template, same k in both — retrieval strategy is the only variable.

Scores are deterministic keyword/source-match checks (0-1), not an LLM judge — see `run_eval.py` docstring. Treat as directional signal; the raw answers in `results/baseline_results.json` / `results/agent_results.json` are the ground truth for actually judging quality.

## Summary

| Metric | Baseline (dense-only) | Agent (hybrid BM25+dense + verification) |
|---|---|---|
| Avg overall score | 0.79 | 0.78 |
| Avg keyword recall | 0.54 | 0.53 |
| Avg source-hit rate | 1.00 | 1.00 |
| Refusal accuracy (out-of-scope case) | 1.00 | 1.00 |
| Avg response time (s) | 21.3 | 49.0 |
| Cases run | 13 | 13 |

Baseline has no verification row/column below -- it's a frozen reproduction of `baseline_code`'s behavior, which never had a verification pass, so it wasn't run with one. Agent's `avg response time` now includes the verification pass (a second full LLM call per question), which is why it's roughly double what it was in the pre-verification run recorded earlier in this file's git history / `CHANGELOG.md`.

### Agent verification verdict distribution

| Verdict | Count (of 13) |
|---|---|
| supported | 10 |
| unsupported | 3 |

## Per-case results

| ID | Difficulty | Question | Baseline score | Agent score | Agent verdict | Baseline latency (s) | Agent latency (s) |
|---|---|---|---|---|---|---|---|
| q01 | easy | What was the outcome of the appeal in LXT Real Estate Broker v SIR Real Estate (CA 005/... | 0.50 | 0.50 | supported | 20.6 | 43.5 |
| q02 | medium | How much did the Court order LXT Real Estate Broker to pay in the Defendant's costs in ... | 0.50 | 0.75 | supported | 18.3 | 48.2 |
| q03 | medium | In Fursa Consulting v Bay Gate Investment (CFI 010/2024), why was the Claimant's applic... | 0.50 | 0.50 | supported | 17.1 | 39.8 |
| q04 | easy | In the arbitration case Ohtli v Onora (ARB 034/2025), what happened to the anti-suit in... | 0.83 | 0.83 | supported | 14.6 | 38.9 |
| q05 | medium | What amount did Okpara originally pay Oralee for visa processing, and what did the Cour... | 1.00 | 0.67 | supported | 29.6 | 44.0 |
| q06 | medium | In Coinmena B.S.C. (C) v Foloosi Technologies Ltd (CFI 067/2025), what percentage of co... | 1.00 | 1.00 | supported | 25.8 | 50.9 |
| q07 | medium | Why did the judge refuse to grant an adjournment of the appeal hearing in Obasi v Orean... | 0.67 | 0.67 | supported | 28.1 | 59.1 |
| q08 | medium | In Oleta v Onesimo [2024] DIFC SCT 454, how much was Oleta owed for her September salar... | 1.00 | 1.00 | supported | 18.1 | 51.6 |
| q09 | hard | In CFI 016/2025 Omar Ben Hallam v Natixis, did Practice Direction No. 1 of 2025 apply r... | 1.00 | 1.00 | unsupported | 22.9 | 54.3 |
| q10 | medium | How much were the Appellants' costs assessed at in CA 004/2025 (Mr Oran and Oaken v Ove... | 1.00 | 1.00 | supported | 20.0 | 45.7 |
| q11 | medium | In CFI 057/2025 Clyde & Co LLP v Union Properties, was the Defendant's Permission to Ap... | 0.50 | 0.50 | unsupported | 18.6 | 50.6 |
| q12 | hard | How much did the Court order the Defendant to pay the Claimant in costs in TCD 001/2024... | 0.75 | 0.75 | supported | 22.0 | 57.9 |
| q13 | hard | What was the outcome of the criminal trial against John Smith in the Dubai Court of Cas... | 1.00 | 1.00 | unsupported | 21.4 | 51.8 |

## Hard case spotlight: q12 (duplicate claim number)

`TCD 001/2024` has two separate cost orders (AED 60,000 and AED 44,000) issued on different dates. This case checks whether each system's retrieval step actually pulls chunks from *both* same-named documents, and whether generation distinguishes them instead of confidently reporting a single wrong figure -- and now, whether verification catches an unflagged conflation after the fact.

- Baseline sources returned: [{'file': 'data/0471e83c1ea18086cfb6b3ff51da6f22b0efee337f10315b2593f782297ccb84.pdf', 'page': '0'}, {'file': 'data/62930da32fa3172edf2f2bbf3da268455bd99a7b5fab34d72358730d8cd5da30.pdf', 'page': '0'}, {'file': 'data/6248961b681ea0deb189f354be0c8286f35974dcdb211c13c921c3dd0e566a6e.pdf', 'page': '0'}]
- Baseline answer: AED 44,000.00.
- Agent sources returned: [{'file': 'data/0471e83c1ea18086cfb6b3ff51da6f22b0efee337f10315b2593f782297ccb84.pdf', 'page': '0'}, {'file': 'data/62930da32fa3172edf2f2bbf3da268455bd99a7b5fab34d72358730d8cd5da30.pdf', 'page': '0'}, {'file': 'data/6248961b681ea0deb189f354be0c8286f35974dcdb211c13c921c3dd0e566a6e.pdf', 'page': '0'}]
- Agent answer: AED 44,000.00.
- Agent verification: {'verdict': 'supported', 'note': 'VERDICT: SUPPORTED\n    The amount is explicitly stated in the source excerpt TCD 001/2024 Architeriors Interior Design (L.L.C) v Emirates National Investment Co (L.L.C) dated January 29, 2026.'}
