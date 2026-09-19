# Agentic AI RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions strictly grounded in the [Agentic AI Ebook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf) by Konverge.ai — built as a take-home assignment for the AI Engineering Intern role.

Built with **LangGraph** for orchestration, **FAISS** as the vector store, **sentence-transformers** for embeddings, and **Groq (Llama 3.1)** as the LLM. Exposed via **FastAPI**, with an optional **Streamlit** UI.

🔗 **Live Demo:** _[add your deployed Streamlit link here]_

🎥 **Video Walkthrough:** https://drive.google.com/file/d/1tmhmHqKg8CRrtVLDd6eNC91pmhHholZh/view?usp=sharing

> A ~2–3 min walkthrough covering: the architecture, a live query, the "I don't know" grounding test, and one of the bugs found + fixed during testing.

---

## Overview

This project ingests a single PDF, indexes it into a vector store, and answers questions about it using a retrieve → generate pipeline orchestrated with LangGraph. Every answer includes:

- **The final answer**
- **The retrieved source chunks** (with page numbers)
- **A confidence score** (average retrieval similarity of the chunks used)

The system is explicitly instructed to answer only from retrieved context, and to say so when a question falls outside the PDF's scope — verified in testing (see [`tests/sample_queries.md`](tests/sample_queries.md)).

## Tech Stack

| Component        | Technology                                  |
|-------------------|---------------------------------------------|
| LLM               | Groq — `openai/gpt-oss-20b`                 |
| Orchestration     | LangGraph                                   |
| Vector Store      | FAISS (local, in-process)                   |
| Embeddings        | sentence-transformers (`all-MiniLM-L6-v2`)  |
| PDF Extraction    | pdfplumber                                  |
| API Framework     | FastAPI + Uvicorn                           |
| UI (optional)     | Streamlit                                   |
| Config            | python-dotenv                               |

> **Note:** FAISS runs fully in-process with file-based persistence — no Docker, no cloud signup, no external services required. Just `pip install`.

## Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com/)

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd rag-chatbot

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your environment
copy .env.example .env
# Then edit .env and paste your Groq API key

# 5. Ingest the PDF (downloads, chunks, embeds, stores — run once)
python -m src.ingest

# 6a. Start the API server
uvicorn src.api:app --reload

# 6b. OR start the Streamlit UI instead
streamlit run src/streamlit_app.py
```

The API starts at `http://127.0.0.1:8000` (interactive docs at `/docs`). The Streamlit UI opens automatically in your browser.

## Usage

### POST /chat

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is Agentic AI?\"}"
```

**Response:**

```json
{
  "answer": "Agentic AI refers to ...",
  "retrieved_chunks": [
    {
      "text": "...",
      "page": 5,
      "score": 0.82
    }
  ],
  "confidence": 0.79
}
```

### GET /health

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "healthy",
  "chunks_in_store": 142
}
```

## Re-ingestion

The ingest script is **idempotent** — running it again skips work if data already exists. To force a fresh re-ingest:

```bash
python -m src.ingest --reset
```

## Sample Queries

Six-plus real Q&A runs against the live system — including two edge cases the pipeline initially got wrong, along with the diagnosis and fix — are documented in [`tests/sample_queries.md`](tests/sample_queries.md). This includes:

- A correctly grounded factual answer
- A correct refusal on an out-of-scope question
- Two cases where the answer was correct despite a low confidence score (demonstrating confidence ≠ correctness)
- A retrieval failure on a broad/enumerative query, and the fix applied
- A false-refusal-despite-high-confidence bug, and the fix applied

## Known Limitations

- **Confidence score is a retrieval-similarity proxy, not a correctness/faithfulness score.** It measures how closely the question's embedding matches the retrieved chunks' embeddings — not whether the generated answer is factually correct. See sample queries for real examples where a correct answer scored low confidence, and vice versa.
- **Fixed top-k retrieval struggles with broad, enumerative questions** (e.g., "list all use cases") where the answer is spread across many non-adjacent chunks. Mitigated with a simple keyword-based dynamic-k adjustment for broad queries — not a full solution, but documented and improved.
- **Page-number citation can occasionally point to a related-but-incorrect section** when two parts of the source document use overlapping vocabulary (e.g., two different sections both describing "perception," "planning," and "learning" in different contexts). This is a known limitation of embedding-based retrieval on documents with repeated terminology across sections.

## Project Structure

```
rag-chatbot/
├── data/                     # Downloaded PDF (auto-created)
├── faiss_index/              # FAISS persistence (auto-created)
├── src/
│   ├── config.py             # Env vars, constants, paths
│   ├── ingest.py             # Download → extract → chunk → embed → store
│   ├── vectorstore.py        # FAISS index + query helpers
│   ├── graph.py              # LangGraph pipeline: retrieve → generate
│   ├── api.py                # FastAPI app
│   └── streamlit_app.py      # Streamlit UI (optional)
├── tests/
│   └── sample_queries.md     # Sample Q&A runs with real output
├── .env.example
├── requirements.txt
├── README.md
└── architecture.md
```

See [architecture.md](architecture.md) for a detailed pipeline explanation and diagram.

## Author

Built by T. Keerthana — [GitHub](https://github.com/Keerthana1367) · [LinkedIn](https://linkedin.com/in/keerthana-tadkal)
