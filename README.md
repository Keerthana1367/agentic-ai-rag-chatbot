# Agentic AI RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions strictly grounded in the [Agentic AI Ebook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf) by Konverge.ai.

Built with **LangGraph** for orchestration, **FAISS** as the vector store, **sentence-transformers** for embeddings, and **Groq (Llama 3.1)** as the LLM. Exposed via **FastAPI**.

## Tech Stack

| Component        | Technology                          |
|------------------|-------------------------------------|
| LLM              | Groq — `openai/gpt-oss-20b`      |
| Orchestration    | LangGraph                           |
| Vector Store     | FAISS (local, in-process)           |
| Embeddings       | sentence-transformers (`all-MiniLM-L6-v2`) |
| PDF Extraction   | pdfplumber                          |
| API Framework    | FastAPI + Uvicorn                   |
| Config           | python-dotenv                       |

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

# 6. Start the API server
uvicorn src.api:app --reload
```

The server starts at `http://127.0.0.1:8000`. Interactive docs are at `/docs`.

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
│   └── api.py                # FastAPI app
├── tests/
│   └── sample_queries.md     # Sample Q&A runs with real output
├── .env.example
├── requirements.txt
├── README.md
└── architecture.md
```

See [architecture.md](architecture.md) for a detailed pipeline explanation and diagram.
