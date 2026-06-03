# RAG Policy Chatbot - Project Timeline

Date: 2026-06-03

---

## Week 1 — Foundations (Heavy Time)

Get the environment working and understand every piece before combining them.

- Set up Docker + Qdrant locally via Docker Compose
- Call the OpenAI API manually (embeddings + chat completion)
- Parse a PDF using PyMuPDF or pdfplumber
- Chunk the parsed text (300–500 tokens, ~50 token overlap, with metadata tags)
- Embed chunks and store them in Qdrant

**Goal:** No chatbot yet — just prove each component works in isolation.

---

## Week 2 — Core RAG Pipeline (Heavy Time)

Wire all the pieces together into a working end-to-end pipeline.

- PDF → chunks → embeddings → Qdrant (ingestion pipeline)
- Question → embed → retrieve top chunks → GPT-4o-mini → answer with citations (query pipeline)
- Test entirely via command line

**Goal:** A working RAG pipeline in the terminal. This is the most critical milestone — if this works, everything after is building on solid ground.

---

## Week 3 — MVP (Tapering Begins)

Wrap the pipeline in an API and tie everything together.

- Build FastAPI endpoints: upload PDF, ask a question
- Add conversation memory for follow-up questions
- Format citation responses (section name, document name, page number)
- Docker Compose to orchestrate everything

**Goal: MVP complete.** A barebones chatbot that accepts a PDF and answers questions about it with citations.

---

## Weeks 4–5 — V1: Multi-Tenant System (2–3 hrs/day)

Build the admin vs. employee model and per-company isolation.

- Per-company Qdrant collections
- Company admin role: upload, replace, and version PDFs
- Employee role: query only
- Simple auth (API keys or company access code)
- Keep old and new PDF versions for comprehensive answers

**Goal:** Multiple companies can use the system with their data fully isolated from each other.

---

## Weeks 6–7 — V2: UI (2–3 hrs/day)

Build a usable interface on top of the API.

- Start with Streamlit (Python-native, fast to ship)
- Admin view: upload/manage PDFs
- Employee view: chat interface with cited answers
- Optionally: migrate to a React frontend for a more polished product

**Goal:** A complete, usable product with a real UI that non-technical users can interact with.

---

## Milestone Summary

| Milestone | Target |
|-----------|--------|
| RAG pipeline working in terminal | End of Week 2 |
| MVP (barebones chatbot) | End of Week 3 |
| V1 (multi-tenant + auth) | End of Week 5 |
| V2 (UI + website) | End of Week 7 |
