"""
query.py — RAG query pipeline
Takes a question, finds the most relevant chunks from Qdrant,
and asks GPT-4o-mini to answer using only those chunks.

Usage:
    python3 src/query.py --collection my_company
"""

import os
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

# --- Config ---
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5  # number of chunks to retrieve per query

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333))
)


# --- Step 1: Embed the question ---
def embed_question(question: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question
    )
    return response.data[0].embedding


# --- Step 2: Retrieve top chunks from Qdrant ---
def retrieve_chunks(question_vector: list[float], collection_name: str) -> list[dict]:
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=question_vector,
        limit=TOP_K,
        with_payload=True,
    )
    chunks = []
    for point in results.points:
        chunks.append({
            "text": point.payload.get("text", ""),
            "source": point.payload.get("source", "unknown"),
            "page": point.payload.get("page", "?"),
            "score": round(point.score, 3),
        })
    return chunks


# --- Step 3: Build prompt and ask GPT-4o-mini ---
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


def ask(question: str, collection_name: str) -> str:
    print(f"\nQuestion: {question}")
    print("Searching policy documents...")

    question_vector = embed_question(question)
    chunks = retrieve_chunks(question_vector, collection_name)

    if not chunks:
        return "No relevant policy content found."

    prompt = build_prompt(question, chunks)

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # deterministic — important for policy answers
    )

    answer = response.choices[0].message.content

    print("\n--- Answer ---")
    print(answer)
    print("\n--- Sources Used ---")
    for chunk in chunks:
        print(f"  {chunk['source']} | Page {chunk['page']} | Relevance score: {chunk['score']}")

    return answer


# --- Conversation loop ---
def chat_loop(collection_name: str):
    print(f"\nPolicy Chatbot ready. Collection: '{collection_name}'")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    conversation_history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not question:
            continue

        # Add previous Q&A to context for follow-up support
        conversation_history.append({"role": "user", "content": question})

        answer = ask(question, collection_name)
        conversation_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, help="Qdrant collection name to query")
    args = parser.parse_args()
    chat_loop(args.collection)
