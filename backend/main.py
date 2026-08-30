import json
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager

# The pipeline modules (src/*) resolve "config/config.yaml" and other paths
# relative to the process's current working directory, so anchor both cwd
# and sys.path to the project root regardless of where uvicorn is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.convert_llama_to_open import OpenVINOLLMLoader
from src.log_setup import setup_logger
from src.vector_db import PDFVectorStore
from src.verifier import verify_answer
from langchain_classic.prompts import PromptTemplate
from langchain_classic.globals import set_verbose
import yaml

set_verbose(False)

logger = setup_logger("backend", "")
logger.info("-----------------------BACKEND STARTED---------------------------")

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

state = {}
sessions = set()
failed_attempts = {}  # client ip -> (count, first_attempt_ts)

MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 300


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("config/config.yaml", "r") as f:
        full_config = yaml.safe_load(f)
    config = full_config["vector_db"]

    state["pin"] = str(full_config["auth"]["pin"])

    logger.info("Loading LLM model")
    loader = OpenVINOLLMLoader(logger)
    llm_model = loader.load_openvino_llm()

    logger.info("Loading vector store")
    db_class = PDFVectorStore(logger)
    db = db_class.read_db()

    state["llm_model"] = llm_model
    state["retriever"] = db_class.get_hybrid_retriever(
        db,
        k=config["k"],
        bm25_weight=config["bm25_weight"],
        dense_weight=config["dense_weight"],
        fetch_k=config["hybrid_fetch_k"],
    )
    logger.info("Backend ready to serve requests")
    yield
    state.clear()


app = FastAPI(title="Legal Document Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    pin: str


class LoginResponse(BaseModel):
    token: str


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    file: str
    page: str


class Verification(BaseModel):
    verdict: str
    note: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    response_time: float
    verification: Verification


def build_prompt(question: str):
    """Retrieve relevant chunks for the question and format the LLM prompt."""
    retriever = state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Model is still loading, please try again shortly.")

    docs = retriever.invoke(question)
    sources = [
        Source(
            file=str(doc.metadata.get("source", "unknown")),
            page=str(doc.metadata.get("page", "N/A")),
        )
        for doc in docs
    ]
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return prompt, sources, context


@app.get("/api/health")
def health():
    return {"status": "ok", "ready": "llm_model" in state}


@app.post("/api/login", response_model=LoginResponse)
def login(request: LoginRequest, http_request: Request):
    client_ip = http_request.client.host if http_request.client else "unknown"
    now = time.time()

    count, first_attempt = failed_attempts.get(client_ip, (0, now))
    if count >= MAX_LOGIN_ATTEMPTS and now - first_attempt < LOGIN_LOCKOUT_SECONDS:
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.")

    if request.pin != state.get("pin"):
        if now - first_attempt >= LOGIN_LOCKOUT_SECONDS:
            count, first_attempt = 0, now
        failed_attempts[client_ip] = (count + 1, first_attempt)
        logger.info(f"Failed login attempt from {client_ip}")
        raise HTTPException(status_code=401, detail="Incorrect PIN.")

    failed_attempts.pop(client_ip, None)
    token = secrets.token_urlsafe(24)
    sessions.add(token)
    return LoginResponse(token=token)


def require_auth(authorization: str | None = Header(default=None)):
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="Please log in with the PIN to continue.")


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Depends(require_auth)])
def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    llm_model = state.get("llm_model")
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, please try again shortly.")

    logger.info(f"Question {question}")
    prompt, sources, context = build_prompt(question)

    start = time.time()
    answer = llm_model.invoke(prompt)
    verification = verify_answer(llm_model, question, answer, context)
    elapsed = time.time() - start

    logger.info(f"Answer {answer}")
    logger.info(f"Verification {verification}")
    return ChatResponse(
        answer=answer,
        sources=sources,
        response_time=elapsed,
        verification=Verification(**verification),
    )


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat/stream", dependencies=[Depends(require_auth)])
def chat_stream(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    llm_model = state.get("llm_model")
    if llm_model is None:
        raise HTTPException(status_code=503, detail="Model is still loading, please try again shortly.")

    logger.info(f"Question {question}")
    prompt, sources, context = build_prompt(question)

    def event_stream():
        start = time.time()
        yield sse("sources", {"sources": [s.model_dump() for s in sources]})

        answer_parts = []
        try:
            for token in llm_model.stream(prompt):
                if not token:
                    continue
                answer_parts.append(token)
                yield sse("token", {"text": token})
        except Exception as e:
            logger.error(f"Streaming failed for question '{question}': {e}")
            yield sse("error", {"detail": "Generation failed. Please try again."})
            return

        answer = "".join(answer_parts)
        logger.info(f"Answer {answer}")

        yield sse("status", {"stage": "verifying"})
        verification = verify_answer(llm_model, question, answer, context)
        logger.info(f"Verification {verification}")
        yield sse("verification", verification)

        elapsed = time.time() - start
        yield sse("done", {"response_time": elapsed})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
