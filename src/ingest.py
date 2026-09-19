"""Ingestion pipeline — download, extract, chunk, embed, and store the PDF.

Run as a script:
    python -m src.ingest          # ingest (skip if data already exists)
    python -m src.ingest --reset  # delete existing data and re-ingest
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pdfplumber
import requests
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    FAISS_DIR,
    PDF_FILENAME,
    PDF_URL,
)
from src.vectorstore import add_chunks, load_index, save_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Step 1: Download ────────────────────────────────────────────────────────


def download_pdf(url: str, dest: Path) -> Path:
    """Download the PDF if it isn't already on disk.

    Args:
        url: Remote URL of the PDF.
        dest: Local directory to save the file in.

    Returns:
        Path to the downloaded (or already-existing) PDF file.

    Raises:
        requests.HTTPError: If the download fails with a non-2xx status.
    """
    dest.mkdir(parents=True, exist_ok=True)
    filepath = dest / PDF_FILENAME

    if filepath.exists():
        logger.info("PDF already exists at %s — skipping download", filepath)
        return filepath

    logger.info("Downloading PDF from %s …", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    filepath.write_bytes(response.content)
    logger.info("Saved PDF to %s (%.1f KB)", filepath, len(response.content) / 1024)
    return filepath


# ── Step 2: Extract text ────────────────────────────────────────────────────


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from each page of the PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of (page_number, page_text) tuples. Page numbers are 1-indexed.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # The PDF has 6 pages of front matter. Offset the page number
            # so the LLM cites the printed page numbers correctly.
            printed_page_num = i - 6 if i > 6 else i
            
            text = page.extract_text() or ""
            
            # Extract tables and format them as readable text blocks
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                table_text += "\n\n[TABLE START]\n"
                for row in table:
                    # Filter out None values and remove internal newlines from cells
                    row_clean = [str(cell).replace('\n', ' ') if cell is not None else "" for cell in row]
                    table_text += " | ".join(row_clean) + "\n"
                table_text += "[TABLE END]\n"
                
            full_text = (text + table_text).strip()
            if full_text:
                pages.append((printed_page_num, full_text))

    logger.info("Extracted text and tables from %d non-empty pages", len(pages))
    return pages


# ── Step 3: Chunk text ──────────────────────────────────────────────────────


def chunk_text(
    pages: list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split page texts into overlapping chunks.

    Overlap matters because a sentence that spans a chunk boundary would be
    cut in half without it.  By repeating the last `overlap` characters of
    one chunk at the start of the next, we preserve context continuity so
    the retriever can still find complete thoughts.

    Args:
        pages: List of (page_number, text) tuples from extract_pages().
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters repeated between consecutive chunks.

    Returns:
        List of dicts with keys: chunk_id, text, page, source.
    """
    chunks: list[dict] = []
    chunk_index = 0

    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text_str = text[start:end]

            chunks.append({
                "chunk_id": f"chunk_{chunk_index}",
                "text": chunk_text_str,
                "page": page_num,
                "source": PDF_FILENAME,
            })
            chunk_index += 1

            # Advance by (chunk_size - overlap) so the next chunk starts
            # `overlap` characters before this chunk ended.
            start += chunk_size - overlap

    logger.info("Created %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, overlap)
    return chunks


# ── Step 4: Embed ───────────────────────────────────────────────────────────


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """Batch-encode texts into embedding vectors.

    We call model.encode() once with the full list instead of one-by-one.
    Both approaches are O(n) in the number of texts, but batching lets the
    model process multiple inputs per forward pass on the GPU/CPU, which
    dramatically reduces wall-clock time (lower constant factor).

    Args:
        texts: Raw text strings to embed.

    Returns:
        Numpy array of shape (n, embedding_dim).
    """
    logger.info("Loading embedding model '%s' …", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    logger.info("Encoding %d texts in a single batch …", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings.astype(np.float32)


# ── Orchestrator ────────────────────────────────────────────────────────────


def ingest(reset: bool = False) -> None:
    """Run the full ingestion pipeline.

    Args:
        reset: If True, delete the existing FAISS index and re-ingest from
               scratch.  If False (default), skip ingestion when the index
               already has data (idempotent).
    """
    # Handle --reset: wipe the persisted FAISS directory
    if reset and FAISS_DIR.exists():
        logger.warning("--reset flag set — deleting FAISS index at %s", FAISS_DIR)
        shutil.rmtree(FAISS_DIR)

    index, metadata = load_index()

    # Idempotency: if data already exists, don't duplicate it
    if index.ntotal > 0:
        logger.info(
            "Index already has %d vectors — skipping ingestion. "
            "Use --reset to re-ingest.",
            index.ntotal,
        )
        return

    # Pipeline: download → extract → chunk → embed → store
    pdf_path = download_pdf(PDF_URL, DATA_DIR)
    pages = extract_pages(pdf_path)
    chunks = chunk_text(pages)

    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts)

    add_chunks(index, metadata, embeddings, chunks)
    save_index(index, metadata)

    logger.info("✓ Ingestion complete — %d chunks stored", len(chunks))


# ── CLI entry point ─────────────────────────────────────────────────────────


def main() -> None:
    """Parse CLI args and run ingestion."""
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI PDF into FAISS")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing data and re-ingest from scratch",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
