import re

from langchain_classic.prompts import PromptTemplate

VERIFICATION_PROMPT = PromptTemplate.from_template(
    """<|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>
    You are a fact-checking assistant for a legal document QA system. You will be shown
    source excerpts, a question, and a draft answer. Check whether the draft answer is
    fully and accurately supported by the source excerpts.

    Respond in exactly this format, nothing else:
    VERDICT: SUPPORTED, UNSUPPORTED, or AMBIGUOUS
    NOTE: one short sentence explaining the verdict

    Use UNSUPPORTED if the draft answer states a fact that is not present in, or
    contradicts, the source excerpts. Also use UNSUPPORTED if the draft answer claims
    the information is missing, unavailable, or not mentioned, but you can find an
    answer to the question by reading the source excerpts yourself -- an incorrect
    refusal is a failure, not a safe answer.
    Read every source excerpt individually before deciding, not just the one the draft
    answer seems to rely on. Use AMBIGUOUS if two or more excerpts give different
    figures, dates, or outcomes for what looks like the same fact, and the draft
    answer picked only one of them without acknowledging the other.
    Use SUPPORTED only if the draft answer is fully backed by the source excerpts,
    does not contradict any excerpt, and does not need a caveat.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Source excerpts:
    {context}

    Question: {question}

    Draft answer: {answer}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(SUPPORTED|UNSUPPORTED|AMBIGUOUS)", re.IGNORECASE)
_NOTE_RE = re.compile(r"NOTE:\s*(.+)", re.IGNORECASE)

# Kept short on purpose: the verdict + one-sentence note never needs more than this,
# and the verification pass is already a second full LLM call on top of the answer
# itself, so capping its length keeps the added latency down.
VERIFICATION_MAX_NEW_TOKENS = 80


def verify_answer(llm_model, question, answer, context):
    """
    Runs a second LLM pass that checks a draft answer against the same source context
    the generator saw, to catch hallucinated facts and unflagged ambiguity (e.g. two
    documents both matching a query with different figures) before the answer reaches
    the user.

    Returns {"verdict": "supported"|"unsupported"|"ambiguous"|"unparsed", "note": str}.
    "unparsed" means the model's response didn't match the expected VERDICT/NOTE
    format -- the raw response is returned as the note in that case rather than
    silently dropped.
    """
    prompt = VERIFICATION_PROMPT.format(context=context, question=question, answer=answer)
    raw = llm_model.invoke(prompt, pipeline_kwargs={"max_new_tokens": VERIFICATION_MAX_NEW_TOKENS})

    verdict_match = _VERDICT_RE.search(raw)
    note_match = _NOTE_RE.search(raw)

    verdict = verdict_match.group(1).lower() if verdict_match else "unparsed"
    note = note_match.group(1).strip() if note_match else raw.strip()

    return {"verdict": verdict, "note": note}
