"""
Test 3: Text chunking
- Takes a block of text and splits it into chunks
- Each chunk has a token count and metadata attached
- Uses tiktoken to count tokens (same tokenizer as OpenAI)
Run: python src/test_chunking.py
"""

import tiktoken

CHUNK_SIZE = 400       # target tokens per chunk
OVERLAP = 50           # tokens to overlap between chunks
ENCODING_MODEL = "text-embedding-3-small"

def get_encoder():
    # text-embedding-3-small uses the cl100k_base encoding
    return tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str, source: str = "unknown", page: int = 0) -> list[dict]:
    encoder = get_encoder()
    tokens = encoder.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)

        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": source,
                "page": page,
                "token_count": len(chunk_tokens),
                "chunk_index": len(chunks),
            }
        })

        start += CHUNK_SIZE - OVERLAP  # move forward with overlap

    return chunks


# --- Test with sample text ---
sample_text = """
Section 3.1 - Vacation Policy
Employees are entitled to 15 days of paid vacation per calendar year.
Vacation days do not carry over to the next year unless approved in writing by HR.
Requests must be submitted at least two weeks in advance.

Section 3.2 - Sick Leave
Employees receive 10 days of paid sick leave per year.
A doctor's note is required for absences exceeding 3 consecutive days.
Unused sick leave does not carry over and is not paid out upon termination.
""" * 20  # repeat to simulate a longer document

chunks = chunk_text(sample_text, source="sample_policy.pdf", page=1)
print(f"Total chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i} ---")
    print(f"Tokens: {chunk['metadata']['token_count']}")
    print(f"Metadata: {chunk['metadata']}")
    print(f"Preview: {chunk['text'][:100]}...")

print("\nChunking test passed.")
