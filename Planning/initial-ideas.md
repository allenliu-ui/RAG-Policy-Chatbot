# RAG Policy Chatbot - Initial Ideas

Date: 2026-06-03

## Goal

Create a chatbot that allows companies to add their terms and conditions, contracts, long documents and allow employees to ask a chatbot questions about the conditions instead of having to comb through long text. 

## Users

- Company admin for posting the pdf, employees for querying
- Am I allowed to do x? How many days off do I get and what are the exceptions? Emergencies?
- The chatbot should provide direct quotes from the pdf as well as extract multiple sources of the pdf to come to the clearest possible conclusion

## Policy Sources

- Terms and conditions or contract PDFs are the main source
- Sources should be privatized to only the employees of that company, need some kind of code/password to ensure this?
- If policies change, the company admin should drop the new pdf into the app so that the chatbot can adjust necessarily. It might also keep old and new versions of the PDF to build a more comprehensive answer. 

## Core Features

- Ask policy questions in plain English
- Retrieve relevant policy excerpts
- Answer with citations or links back to source material
- Flag uncertainty instead of guessing

## Open Questions

- What type of policies is this for?
   - company and work policies
- Should it run locally, on a website, or inside Slack/Discord/Telegram?
    -For our start, lets try to do the most optimal method
- Should users be able to upload documents?
    -only company admin, not the employees themselves
- Do answers need citations every time?
    -Yes the answers should always strive for citations if possible
- Should it remember previous conversation context?
     -Yes it should do this to allow for follow up questions

## Tech Stack Decisions

- **Backend**: Python + FastAPI
- **PDF Parsing**: PyMuPDF or pdfplumber
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT-4o-mini (cost-efficient, sufficient for retrieval-based Q&A)
- **Vector Database**: Qdrant (Docker) — per-company separate collections for clean multi-tenant isolation
- **Local Orchestration**: Docker Compose
- **Chunking Strategy**: By paragraph/clause, 300–500 tokens per chunk, ~50 token overlap; each chunk tagged with section name, document name, and page number as metadata for citations

## Notes
   Q1: What is the main goal of this project?
      Learn a new concept (RAG and AI Chatbot) First try at turning an idea into a real product. Follow a timeline and come through with it
    Q2: Milestones of Functionality
    MVP: A chatbot that can take in a pdf of terms and conditions and allow the user to ask questions about the PDF. Just the barebones Chatbot
    V1: Develop a Employer, Employee side to create system (potential cybersecurity concepts?)
    V2: Develop UI, website 


