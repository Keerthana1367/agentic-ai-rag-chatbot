"""FastAPI application — exposes the RAG pipeline over HTTP.

Run with:
    uvicorn src.api:app --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import run_query
from src.vectorstore import load_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Pydantic schemas ────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming question from the user."""

    question: str = Field(
        ...,
        min_length=1,
        description="The question to ask about the Agentic AI ebook",
        examples=["What is Agentic AI?"],
    )


class ChunkResponse(BaseModel):
    """A single retrieved chunk returned alongside the answer."""

    text: str
    page: int
    score: float


class ChatResponse(BaseModel):
    """Full response: the LLM answer, supporting chunks, and confidence."""

    answer: str
    retrieved_chunks: list[ChunkResponse]
    confidence: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    chunks_in_store: int


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Verify FAISS index is loaded and has data on startup."""
    index, _ = load_index()
    if index.ntotal == 0:
        logger.warning(
            "FAISS index is empty — run `python -m src.ingest` first"
        )
    else:
        logger.info("FAISS index ready with %d vectors", index.ntotal)
    yield


# ── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="Ask questions about the Agentic AI ebook — answers are grounded strictly in the PDF content.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permissive for local testing
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check that the API and FAISS index are operational."""
    try:
        index, _ = load_index()
        count = index.ntotal
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        raise HTTPException(status_code=503, detail="FAISS index is unavailable") from exc

    return HealthResponse(status="healthy", chunks_in_store=count)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using the RAG pipeline.

    The question is embedded, matched against stored PDF chunks, and then
    answered by the Groq LLM using only the retrieved context.
    """
    logger.info("POST /chat — question: '%s'", request.question)

    try:
        result = run_query(request.question)
    except ValueError as exc:
        # Raised when GROQ_API_KEY is missing
        logger.error("Configuration error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        # Catch Groq / FAISS / network errors and return a clean 502
        # instead of leaking a raw stack trace to the client.
        logger.error("Pipeline error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=502,
            detail="An error occurred while processing your question. Please try again.",
        ) from exc

    chunks = [
        ChunkResponse(text=c["text"], page=c["page"], score=c["score"])
        for c in result["retrieved_chunks"]
    ]

    return ChatResponse(
        answer=result["answer"],
        retrieved_chunks=chunks,
        confidence=result["confidence"],
    )
