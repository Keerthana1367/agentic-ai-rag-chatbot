"""Streamlit chat UI — a clean frontend for the RAG chatbot.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st

from src.graph import run_query
from src.vectorstore import load_index

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helper ──────────────────────────────────────────────────────────────────


def show_sources(chunks: list[dict], confidence: float) -> None:
    """Render the retrieved source chunks and confidence score."""
    if not chunks:
        return

    # Confidence indicator
    conf_pct = int(confidence * 100)
    if confidence >= 0.7:
        conf_color = "#4caf50"
    elif confidence >= 0.4:
        conf_color = "#ff9800"
    else:
        conf_color = "#f44336"

    st.markdown(
        f"**Confidence:** {conf_pct}%"
        f'<div class="confidence-bar" style="width:{conf_pct}%;background:{conf_color};"></div>',
        unsafe_allow_html=True,
    )

    with st.expander(f"📄 View {len(chunks)} source chunks"):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(
                f"**[{i}]** Page {chunk['page']} · "
                f"Score: {chunk['score']:.2f}"
            )
            st.text(chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""))
            if i < len(chunks):
                st.divider()


# ── Header ──────────────────────────────────────────────────────────────────

st.title("🤖 Agentic AI Chatbot")
st.caption("Ask anything about the Agentic AI ebook — answers are grounded strictly in the PDF.")

# ── Sidebar: index status ───────────────────────────────────────────────────

with st.sidebar:
    st.header("📊 Index Status")
    try:
        index, _ = load_index()
        chunk_count = index.ntotal
        if chunk_count > 0:
            st.success(f"✅ {chunk_count} chunks indexed")
        else:
            st.warning("⚠️ Index is empty — run `python -m src.ingest`")
    except Exception:
        st.error("❌ Could not load FAISS index")
        chunk_count = 0

    st.divider()
    st.header("ℹ️ About")
    st.markdown(
        "This chatbot uses **RAG** (Retrieval-Augmented Generation) to answer "
        "questions based solely on the [Agentic AI Ebook]"
        "(https://konverge.ai/pdf/Ebook-Agentic-AI.pdf).\n\n"
        "**Stack:** LangGraph · FAISS · Groq (Llama 3.1) · sentence-transformers"
    )

# ── Chat history ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and "chunks" in msg:
            show_sources(msg["chunks"], msg.get("confidence", 0))

# ── Chat input ──────────────────────────────────────────────────────────────

if question := st.chat_input("Ask a question about Agentic AI…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = run_query(question)
                answer = result["answer"]
                chunks = result["retrieved_chunks"]
                confidence = result["confidence"]

                st.markdown(answer)
                show_sources(chunks, confidence)

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "chunks": chunks,
                    "confidence": confidence,
                })

            except ValueError as exc:
                st.error(f"⚠️ Configuration error: {exc}")
            except Exception as exc:
                st.error(f"❌ Something went wrong: {exc}")
