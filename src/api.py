"""
api.py — FastAPI wrapper for the RAG pipeline (Week 4: multi-tenant auth)

Endpoints:
    GET  /health              — Check API and Qdrant connectivity (public)

    POST /ingest              — Upload a PDF; collection derived from API key
    POST /query               — Ask a question; collection derived from API key

    POST /admin/keys          — Create a new tenant API key  [X-Admin-Key]
    GET  /admin/keys          — List all keys (redacted)     [X-Admin-Key]
    POST /admin/keys/revoke   — Revoke a key                 [X-Admin-Key]

Auth model:
    Tenant endpoints use the header:  X-API-Key: rag_<hex>
    Admin endpoints use the header:   X-Admin-Key: <ADMIN_SECRET from .env>

Usage:
    uvicorn src.api:app --reload
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, File, Form, Header, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import fitz  # PyMuPDF
import tiktoken
import uuid

from src.auth import create_key, validate_key, revoke_key, list_keys

load_dotenv()

# --- Config ---
CHUNK_SIZE = 400
OVERLAP = 50
VECTOR_SIZE = 1536
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333))
)
encoder = tiktoken.get_encoding("cl100k_base")

app = FastAPI(
    title="RAG Policy Chatbot API",
    description="Upload PDF policy documents and query them with natural language.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Security schemes (appear in /docs) ---
tenant_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
admin_key_scheme = APIKeyHeader(name="X-Admin-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Auth dependencies
# ---------------------------------------------------------------------------

def require_tenant(x_api_key: str | None = Depends(tenant_key_scheme)) -> dict:
    """
    FastAPI dependency: validates X-API-Key and returns the tenant info dict.
    Raises 401 if missing or invalid.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
    info = validate_key(x_api_key)
    if info is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key.")
    return info


def require_admin(x_admin_key: str | None = Depends(admin_key_scheme)) -> None:
    """
    FastAPI dependency: validates X-Admin-Key against ADMIN_SECRET.
    Raises 401/403 as appropriate.
    """
    if not x_admin_key:
        raise HTTPException(status_code=401, detail="Missing X-Admin-Key header.")
    if not ADMIN_SECRET:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_SECRET is not configured on this server."
        )
    if x_admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    message: str
    tenant: str
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
    tenant: str
    collection: str


class CreateKeyRequest(BaseModel):
    tenant: str
    collection: str


class CreateKeyResponse(BaseModel):
    api_key: str
    tenant: str
    collection: str
    message: str


class RevokeKeyRequest(BaseModel):
    api_key: str


class KeyInfo(BaseModel):
    key_hint: str
    tenant: str
    collection: str
    created_at: str
    active: bool


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

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


def retrieve_chunks(question_vector: list[float], collection_name: str) -> list[dict]:
    if not qdrant_client.collection_exists(collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection_name}' not found. Ingest a PDF first."
        )
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


# ---------------------------------------------------------------------------
# Public endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Check API and Qdrant connectivity."""
    try:
        qdrant_client.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant unreachable: {e}")


# ---------------------------------------------------------------------------
# Tenant endpoints (require X-API-Key)
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(..., description="PDF file to ingest"),
    tenant_info: dict = Depends(require_tenant),
):
    """
    Upload a PDF and store its embeddings in the tenant's Qdrant collection.
    The collection is determined by the API key — callers cannot choose it.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    collection = tenant_info["collection"]
    tenant = tenant_info["tenant"]

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
        tenant=tenant,
        collection=collection,
        chunks_stored=len(chunks),
        filename=file.filename,
    )


@app.post("/query", response_model=QueryResponse)
def query(
    question: str = Form(..., description="The question to ask"),
    tenant_info: dict = Depends(require_tenant),
):
    """
    Ask a question and get a cited answer from the tenant's ingested documents.
    The collection is determined by the API key.
    """
    collection = tenant_info["collection"]
    tenant = tenant_info["tenant"]

    question_vector = openai_client.embeddings.create(
        model=EMBEDDING_MODEL, input=question
    ).data[0].embedding

    chunks = retrieve_chunks(question_vector, collection)

    if not chunks:
        return QueryResponse(
            answer="No relevant content found.",
            sources=[],
            tenant=tenant,
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
        tenant=tenant,
        collection=collection,
    )


# ---------------------------------------------------------------------------
# Admin endpoints (require X-Admin-Key)
# ---------------------------------------------------------------------------

@app.post("/admin/keys", response_model=CreateKeyResponse, dependencies=[Depends(require_admin)])
def admin_create_key(body: CreateKeyRequest):
    """
    Create a new API key scoped to the given tenant name and Qdrant collection.
    The returned key is shown only once — store it immediately.
    """
    api_key = create_key(tenant=body.tenant, collection=body.collection)
    return CreateKeyResponse(
        api_key=api_key,
        tenant=body.tenant,
        collection=body.collection,
        message="Store this key securely — it will not be shown again.",
    )


@app.get("/admin/keys", response_model=list[KeyInfo], dependencies=[Depends(require_admin)])
def admin_list_keys():
    """List all API keys (key values redacted to last 8 chars)."""
    return list_keys()


@app.post("/admin/keys/revoke", dependencies=[Depends(require_admin)])
def admin_revoke_key(body: RevokeKeyRequest):
    """Revoke an API key so it can no longer authenticate."""
    found = revoke_key(body.api_key)
    if not found:
        raise HTTPException(status_code=404, detail="Key not found.")
    return {"message": "Key revoked successfully."}
