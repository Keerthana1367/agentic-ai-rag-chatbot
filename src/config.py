"""Configuration — environment variables, constants, and paths.

Loads settings from a .env file and exposes them as module-level constants
so every other module can just `from src.config import X`.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_DIR = PROJECT_ROOT / "faiss_index"

# ── PDF source ───────────────────────────────────────────────────────────────
PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
PDF_FILENAME = "Ebook-Agentic-AI.pdf"

# ── Embedding model ─────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # output dimension for all-MiniLM-L6-v2

# ── FAISS index ──────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
FAISS_METADATA_PATH = FAISS_DIR / "metadata.json"

# ── Chunking parameters ─────────────────────────────────────────────────────
CHUNK_SIZE = 700      # characters per chunk
CHUNK_OVERLAP = 100   # overlap between consecutive chunks

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 4

# ── Groq LLM ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-20b"
