"""
ingest.py — PDF ingestion pipeline
Takes a PDF, chunks it, embeds each chunk, and stores it in Qdrant.

Usage:
    python3 src/ingest.py --pdf data/your_policy.pdf --collection my_company
"""

import os
import argparse
import uuid
import fitz  # PyMuPDF
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# --- Config ---
CHUNK_SIZE = 400      # tokens per chunk
OVERLAP = 50          # token overlap between chunks
VECTOR_SIZE = 1536    # dimensions for text-embedding-3-small
EMBEDDING_MODEL = "text-embedding-3-small"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333))
)
encoder = tiktoken.get_encoding("cl100k_base")


# --- Step 1: Extract text from PDF ---
def extract_text(pdf_path: str) -> list[dict]:
    """Returns a list of {page, text} dicts, one per page."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})
    print(f"  Extracted {len(pages)} pages from {pdf_path}")
    return pages


# --- Step 2: Chunk text ---
def chunk_pages(pages: list[dict], source: str) -> list[dict]:
    """Splits page text into overlapping token chunks with metadata."""
    chunks = []
    for page_data in pages:
        tokens = encoder.encode(page_data["text"])
        start = 0
        while start < len(tokens):
            end = start + CHUNK_SIZE
            chunk_tokens = tokens[start:end]
            chunk_text = encoder.decode(chunk_tokens)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "page": page_data["page"],
                    "token_count": len(chunk_tokens),
                    "chunk_index": len(chunks),
                }
            })
            start += CHUNK_SIZE - OVERLAP
    print(f"  Created {len(chunks)} chunks")
    return chunks


# --- Step 3: Embed chunks ---
def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Adds an embedding vector to each chunk."""
    texts = [chunk["text"] for chunk in chunks]

    # OpenAI allows up to 2048 inputs per request, batch to be safe
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])

    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    print(f"  Embedded {len(chunks)} chunks")
    return chunks


# --- Step 4: Store in Qdrant ---
def store_in_qdrant(chunks: list[dict], collection_name: str):
    """Creates collection if needed and upserts all chunks."""
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  Created collection '{collection_name}'")
    else:
        print(f"  Collection '{collection_name}' already exists, adding to it")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk["embedding"],
            payload={
                "text": chunk["text"],
                **chunk["metadata"]
            }
        )
        for chunk in chunks
    ]

    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"  Stored {len(points)} vectors in Qdrant")


# --- Main ---
def ingest(pdf_path: str, collection_name: str):
    print(f"\nIngesting '{pdf_path}' into collection '{collection_name}'...")
    pages = extract_text(pdf_path)
    chunks = chunk_pages(pages, source=os.path.basename(pdf_path))
    chunks = embed_chunks(chunks)
    store_in_qdrant(chunks, collection_name)
    print(f"\nDone. {len(chunks)} chunks ready for querying.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--collection", required=True, help="Qdrant collection name (one per company)")
    args = parser.parse_args()
    ingest(args.pdf, args.collection)
