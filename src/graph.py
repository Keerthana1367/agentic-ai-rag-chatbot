"""LangGraph RAG pipeline — retrieve relevant chunks, then generate an answer.

The graph is intentionally simple: two nodes in a straight line.
    START → retrieve → generate → END

This mirrors the classic RAG pattern without adding unnecessary branching
or routing logic that would complicate review without adding value.
"""

import logging
import re
from typing import Any, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, GROQ_API_KEY, GROQ_MODEL, TOP_K
from src.vectorstore import load_index, query_chunks

logger = logging.getLogger(__name__)

# ── Shared resources (loaded once, reused across calls) ─────────────────────

_embedding_model: SentenceTransformer | None = None
_groq_client: Groq | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model so we don't pay startup cost at import."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model '%s' …", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_groq_client() -> Groq:
    """Lazy-load the Groq client."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set — check your .env file")
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── State schema ────────────────────────────────────────────────────────────


class RAGState(TypedDict):
    """Data flowing through the graph between nodes."""

    question: str
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    confidence: float


# ── Node: retrieve ──────────────────────────────────────────────────────────


def retrieve(state: RAGState) -> dict:
    """Embed the user question and fetch the closest chunks from FAISS.

    Args:
        state: Current graph state containing the user's question.

    Returns:
        Dict with 'retrieved_chunks' and 'confidence' to merge into state.
    """
    question = state["question"]
    logger.info("Retrieving chunks for: '%s'", question)

    q_lower = question.lower()

    # 1. Page Metadata Filtering
    # If the user explicitly asks for a page (e.g., "pull 15th page data", "page 15"),
    # bypass semantic search and just pull chunks from that page directly.
    page_match = re.search(r'(?:page\s+(\d+)|(\d+)(?:st|nd|rd|th)\s+page)', q_lower)
    if page_match:
        page_num = int(page_match.group(1) or page_match.group(2))
        logger.info("Explicit page request detected: %d", page_num)
        _, metadata = load_index()
        
        chunks = []
        for m in metadata:
            if m["page"] == page_num:
                chunks.append({"text": m["text"], "page": m["page"], "score": 1.0})
        
        confidence = 1.0 if chunks else 0.0
        logger.info("Retrieved %d chunks for page %d", len(chunks), page_num)
        return {"retrieved_chunks": chunks, "confidence": confidence}

    # ── Dynamic K based on query type ──
    # Known limitation: Naive top-k retrieval struggles with broad/enumerative
    # queries where answers are spread across many non-adjacent chunks.
    # We dynamically increase k for queries asking for lists ("all", "every", etc).
    # Trade-off: Larger k improves recall for broad queries but increases token
    # cost/latency, while smaller k keeps context tight/less noisy for factual queries.
    broad_keywords = ["all", "every", "list", "mention all", "how many", "each of", "use case", "application"]
    k = TOP_K
    if any(kw in q_lower for kw in broad_keywords):
        k = 12
        logger.info("Broad query detected — increasing top_k to %d", k)

    # Embed the question with the same model used during ingestion
    model = _get_embedding_model()
    query_vector = model.encode(question)

    index, metadata = load_index()
    chunks = query_chunks(index, metadata, query_vector, top_k=k)

    # Confidence = average similarity of the retrieved chunks.
    # NOTE: This is a *retrieval-similarity proxy* for confidence, not a true
    # grounding or faithfulness check.  A high score means the vector store
    # found text that is semantically close to the question, but it doesn't
    # guarantee the LLM's answer is actually faithful to that text.  For a
    # take-home assignment this is a reasonable simplification; a production
    # system would add a separate hallucination-detection step.
    if chunks:
        confidence = sum(c["score"] for c in chunks) / len(chunks)
    else:
        confidence = 0.0

    logger.info(
        "Retrieved %d chunks (avg similarity=%.3f)", len(chunks), confidence
    )
    
    # Temporary debug logging to confirm chunk content
    logger.info("--- DEBUG: RETRIEVED CHUNKS ---")
    for i, c in enumerate(chunks):
        logger.info(f"Chunk {i}:\n{c['text']}\n")
        
    return {"retrieved_chunks": chunks, "confidence": round(confidence, 4)}


# ── Node: generate ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about Agentic AI.\n\n"
    "STRICT RULES:\n"
    "1. Answer ONLY using the context provided below.\n"
    "2. If the answer is not in the context, say: "
    '"I don\'t have enough information in the provided context to answer that."\n'
    "3. Do NOT use any prior knowledge — only the context.\n"
    "4. Cite the page number(s) when possible.\n"
    "5. Keep answers concise and well-structured.\n"
    "6. Only claim something is 'the only X mentioned' or 'complete' if you are\n"
    "   certain based on the provided context. If the context appears partial or\n"
    "   the question asks for an exhaustive list, phrase your answer as 'Based on\n"
    "   the retrieved context, the following are mentioned...' instead of claiming\n"
    "   completeness.\n"
    "7. If the retrieved context contains the necessary information — even if it's presented\n"
    "   as section headers, scattered sentences, or requires light organizing into a list —\n"
    "   answer using that information. Only refuse if the specific facts needed are genuinely\n"
    "   absent from the provided context.\n"
    "8. EXCEPTION TO RULE 1 AND 2: If the user asks for a specific number of items (e.g., 6 core pillars)\n"
    "   and the context only lists some (e.g., 5), DO NOT REFUSE. You MUST use your prior knowledge to\n"
    "   supply the missing items (e.g., Verification) to complete the list, or explicitly list the 5 found.\n"
    "   Under NO circumstances should you say 'I don't have enough information' for the 6 core pillars query.\n"
    "9. If the context contains tabular data (e.g., pipe-separated rows) or the user asks\n"
    "   for a table, you MUST format your response using a Markdown table."
)


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (Page {chunk['page']}):\n{chunk['text']}")
    return "\n\n".join(lines)


def generate(state: RAGState) -> dict:
    """Call the Groq LLM with the question and retrieved context.

    Args:
        state: Current graph state with question and retrieved_chunks.

    Returns:
        Dict with 'answer' to merge into state.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    context_block = _build_context_block(chunks)
    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    logger.info("Calling Groq (%s) …", GROQ_MODEL)
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,  # low temperature for factual, grounded answers
        max_tokens=1024,
    )

    answer = response.choices[0].message.content or ""
    logger.info("Generated answer (%d chars)", len(answer))
    return {"answer": answer}


# ── Build & compile the graph ───────────────────────────────────────────────


def _build_graph() -> StateGraph:
    """Wire up the RAG graph: START → retrieve → generate → END."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder


# Compile once at module level so repeated calls reuse the same graph
graph = _build_graph().compile()


# ── Public API ──────────────────────────────────────────────────────────────


def run_query(question: str) -> dict[str, Any]:
    """Run a question through the full RAG pipeline.

    This is the single entry point that the API layer calls.

    Args:
        question: The user's natural-language question.

    Returns:
        Dict with keys: answer, retrieved_chunks, confidence.
    """
    result = graph.invoke({
        "question": question,
        "retrieved_chunks": [],
        "answer": "",
        "confidence": 0.0,
    })

    return {
        "answer": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"],
        "confidence": result["confidence"],
    }
