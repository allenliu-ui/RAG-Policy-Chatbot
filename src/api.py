"""
api.py — FastAPI wrapper for the RAG pipeline

Endpoints:
    POST /ingest   — Upload a PDF and store its embeddings in Qdrant
    POST /query    — Ask a question against a collection, get a cited answer
    GET  /health   — Check that the API and Qdrant are reachable

Usage:
    uvicorn src.api:app --reload
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import fitz  # PyMuPDF
import tiktoken
import uuid

load_dotenv()

# --- Config ---
CHUNK_SIZE = 400
OVERLAP = 50
VECTOR_SIZE = 1536
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333))
)
encoder = tiktoken.get_encoding("cl100k_base")

app = FastAPI(
    title="RAG Policy Chatbot API",
    description="Upload PDF policy documents and query them with natural language.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Response models ---
class IngestResponse(BaseModel):
    message: str
    collection: str
    chunks_stored: int
    filename: str


class Source(BaseModel):
    source: str
    page: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    collection: str


# --- Ingest helpers (same logic as ingest.py) ---
def extract_text(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})
    return pages


def chunk_pages(pages: list[dict], source: str) -> list[dict]:
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
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def store_in_qdrant(chunks: list[dict], collection_name: str):
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk["embedding"],
            payload={"text": chunk["text"], **chunk["metadata"]}
        )
        for chunk in chunks
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)


# --- Query helpers (same logic as query.py) ---
def retrieve_chunks(question_vector: list[float], collection_name: str) -> list[dict]:
    if not qdrant_client.collection_exists(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found. Ingest a PDF first.")
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=question_vector,
        limit=TOP_K,
        with_payload=True,
    )
    return [
        {
            "text": p.payload.get("text", ""),
            "source": p.payload.get("source", "unknown"),
            "page": p.payload.get("page", 0),
            "score": round(p.score, 3),
        }
        for p in results.points
    ]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}\n"
    return f"""You are a helpful assistant that answers questions about company policies.
Use ONLY the policy excerpts provided below to answer the question.
If the answer is not clearly stated in the excerpts, say "I could not find a clear answer in the provided policy documents."
Always cite the source and page number for each point you make.

Policy Excerpts:
{context}

Question: {question}

Answer:"""


# --- Endpoints ---
@app.get("/health")
def health():
    """Check API and Qdrant connectivity."""
    try:
        qdrant_client.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {e}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(..., description="PDF file to ingest"),
    collection: str = Form(..., description="Collection name (one per company/tenant)"),
):
    """Upload a PDF and store its embeddings in Qdrant."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        pages = extract_text(tmp_path)
        if not pages:
            raise HTTPException(status_code=422, detail="No readable text found in PDF.")

        chunks = chunk_pages(pages, source=file.filename)
        embeddings = embed_texts([c["text"] for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        store_in_qdrant(chunks, collection)
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        message="PDF ingested successfully.",
        collection=collection,
        chunks_stored=len(chunks),
        filename=file.filename,
    )


@app.post("/query", response_model=QueryResponse)
def query(
    question: str = Form(..., description="The question to ask"),
    collection: str = Form(..., description="Collection name to search"),
):
    """Ask a question and get a cited answer from the ingested documents."""
    question_vector = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=question
    ).data[0].embedding

    chunks = retrieve_chunks(question_vector, collection)

    if not chunks:
        return QueryResponse(
            answer="No relevant content found.",
            sources=[],
            collection=collection,
        )

    prompt = build_prompt(question, chunks)
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content

    return QueryResponse(
        answer=answer,
        sources=[Source(source=c["source"], page=c["page"], score=c["score"]) for c in chunks],
        collection=collection,
    )
