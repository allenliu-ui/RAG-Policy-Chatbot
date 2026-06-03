"""
Test 4: Qdrant connection
- Connects to local Qdrant instance
- Creates a test collection
- Inserts a dummy vector
- Queries it back
Run: python src/test_qdrant.py
Make sure Docker is running: docker compose up -d
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "test_collection"
VECTOR_SIZE = 1536  # dimensions for text-embedding-3-small

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# --- Create collection ---
print("Creating test collection...")
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
print(f"Collection '{COLLECTION_NAME}' created.")

# --- Insert a dummy vector ---
print("Inserting dummy vector...")
dummy_vector = [0.1] * VECTOR_SIZE
client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=1,
            vector=dummy_vector,
            payload={"text": "This is a test chunk.", "source": "test.pdf", "page": 1}
        )
    ]
)
print("Vector inserted.")

# --- Query it back ---
print("Querying collection...")
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=dummy_vector,
    limit=1,
)
print(f"Query result: {results.points[0].payload}")
print("\nQdrant test passed.")
