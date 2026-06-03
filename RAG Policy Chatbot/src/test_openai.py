"""
Test 1: OpenAI API connection
- Sends a simple chat message
- Generates an embedding for a test string
Run: python src/test_openai.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Test chat completion ---
print("Testing chat completion...")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print("Chat response:", response.choices[0].message.content)

# --- Test embeddings ---
print("\nTesting embeddings...")
embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input="This is a test policy document sentence."
)
vector = embedding_response.data[0].embedding
print(f"Embedding generated. Dimensions: {len(vector)}")
print("OpenAI tests passed.")
