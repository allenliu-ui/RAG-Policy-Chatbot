"""
streamlit_app.py — RAG Policy Chatbot UI

Connects to the FastAPI backend (src/api.py) via HTTP.
Run with:
    python3 -m streamlit run streamlit_app.py

Requires the FastAPI server to be running:
    python3 -m uvicorn src.api:app --reload
"""

import streamlit as st
import requests
from typing import Optional, List

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"
MAX_HISTORY_TURNS = 5   # exchanges to send to the API for memory

st.set_page_config(
    page_title="Policy Chatbot",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for key, default in {
    "api_key": "",
    "tenant_info": None,
    "messages": [],
    "ingest_status": None,
    "documents": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def verify_key(api_key: str) -> Optional[dict]:
    """Call /me to validate the key and return tenant info."""
    try:
        resp = requests.get(
            f"{API_BASE}/me",
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except requests.ConnectionError:
        st.error("Cannot reach the API server. Is it running?")
        return None


def call_ingest(api_key: str, uploaded_file) -> dict:
    resp = requests.post(
        f"{API_BASE}/ingest",
        headers={"X-API-Key": api_key},
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def call_query(api_key: str, question: str, history: List[dict]) -> dict:
    resp = requests.post(
        f"{API_BASE}/query",
        headers={"X-API-Key": api_key},
        json={"question": question, "history": history},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def call_list_documents(api_key: str) -> List[str]:
    try:
        resp = requests.get(
            f"{API_BASE}/documents",
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("documents", [])
    except Exception:
        pass
    return []


def build_history(messages: List[dict]) -> List[dict]:
    """
    Convert session messages to API history format.
    Sends the last MAX_HISTORY_TURNS exchanges (user + assistant pairs).
    """
    history = []
    for msg in messages[-(MAX_HISTORY_TURNS * 2):]:
        history.append({"role": msg["role"], "content": msg["content"]})
    return history


def confidence_badge(score: float) -> str:
    """Color-coded dot based on relevance score."""
    if score >= 0.80:
        return "🟢"
    elif score >= 0.60:
        return "🟡"
    else:
        return "🔴"


def confidence_label(score: float) -> str:
    if score >= 0.80:
        return "High"
    elif score >= 0.60:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📋 Policy Chatbot")
    st.markdown("---")

    # --- API Key ---
    st.subheader("🔑 API Key")
    key_input = st.text_input(
        "Enter your API key",
        type="password",
        value=st.session_state.api_key,
        placeholder="rag_...",
    )

    if key_input != st.session_state.api_key:
        st.session_state.api_key = key_input
        st.session_state.tenant_info = None
        st.session_state.messages = []
        st.session_state.documents = []

    if st.session_state.api_key and st.session_state.tenant_info is None:
        with st.spinner("Verifying key..."):
            info = verify_key(st.session_state.api_key)
        if info:
            st.session_state.tenant_info = info
            st.session_state.documents = call_list_documents(st.session_state.api_key)
        else:
            st.error("Invalid or revoked API key.")

    if st.session_state.tenant_info:
        info = st.session_state.tenant_info
        st.success("✅ Connected")
        st.caption(f"**Tenant:** {info['tenant']}")
        st.caption(f"**Collection:** {info['collection']}")

    st.markdown("---")

    # --- PDF Ingest ---
    st.subheader("📄 Ingest a PDF")

    if not st.session_state.tenant_info:
        st.info("Enter a valid API key above to enable ingestion.")
    else:
        uploaded = st.file_uploader(
            "Upload a policy PDF",
            type=["pdf"],
            help="The PDF will be chunked, embedded, and stored in your collection.",
        )

        if uploaded:
            if st.button("⬆️ Ingest PDF", use_container_width=True):
                with st.spinner(f"Ingesting {uploaded.name}…"):
                    try:
                        result = call_ingest(st.session_state.api_key, uploaded)
                        st.session_state.ingest_status = {
                            "ok": True,
                            "msg": f"✅ {result['chunks_stored']} chunks stored from **{result['filename']}**.",
                        }
                        # Refresh document list
                        st.session_state.documents = call_list_documents(st.session_state.api_key)
                    except requests.HTTPError as e:
                        detail = e.response.json().get("detail", str(e))
                        st.session_state.ingest_status = {"ok": False, "msg": f"❌ {detail}"}
                    except Exception as e:
                        st.session_state.ingest_status = {"ok": False, "msg": f"❌ {e}"}

        if st.session_state.ingest_status:
            s = st.session_state.ingest_status
            if s["ok"]:
                st.success(s["msg"])
            else:
                st.error(s["msg"])

    st.markdown("---")

    # --- Ingested Documents ---
    st.subheader("📂 Ingested Documents")

    if not st.session_state.tenant_info:
        st.caption("Connect to see available documents.")
    elif not st.session_state.documents:
        st.caption("No documents ingested yet.")
    else:
        for doc in st.session_state.documents:
            st.caption(f"• {doc}")

    if st.session_state.tenant_info:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.documents = call_list_documents(st.session_state.api_key)
            st.rerun()

    st.markdown("---")

    # --- Clear chat ---
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.header("Ask a policy question")

if not st.session_state.tenant_info:
    st.info("👈 Enter your API key in the sidebar to get started.")
    st.stop()

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    badge = confidence_badge(s["score"])
                    label = confidence_label(s["score"])
                    st.markdown(
                        f"{badge} **{s['source']}** — Page {s['page']} &nbsp; "
                        f"`{label} relevance ({s['score']})`"
                    )

# Chat input
question = st.chat_input("Ask a question about your policy documents…")

if question:
    # Build history before appending the new message
    history = build_history(st.session_state.messages)

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents…"):
            try:
                result = call_query(st.session_state.api_key, question, history)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for s in sources:
                            badge = confidence_badge(s["score"])
                            label = confidence_label(s["score"])
                            st.markdown(
                                f"{badge} **{s['source']}** — Page {s['page']} &nbsp; "
                                f"`{label} relevance ({s['score']})`"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.HTTPError as e:
                try:
                    detail = e.response.json().get("detail", str(e))
                except Exception:
                    detail = str(e)
                err = f"⚠️ API error: {detail}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
            except requests.ConnectionError:
                err = "⚠️ Cannot reach the API. Is the server running?"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
