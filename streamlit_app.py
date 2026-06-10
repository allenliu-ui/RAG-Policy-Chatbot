"""
streamlit_app.py — Week 5 UI for the RAG Policy Chatbot

Connects to the FastAPI backend (src/api.py) via HTTP.
Run with:
    streamlit run streamlit_app.py

Requires the FastAPI server to be running:
    uvicorn src.api:app --reload
"""

import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Policy Chatbot",
    page_icon="📋",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "tenant_info" not in st.session_state:
    st.session_state.tenant_info = None   # dict with tenant + collection once key is validated
if "messages" not in st.session_state:
    st.session_state.messages = []        # list of {"role": "user"|"assistant", "content": str, "sources": list}
if "ingest_status" not in st.session_state:
    st.session_state.ingest_status = None


# ---------------------------------------------------------------------------
# Helper: validate key by calling /health is not enough — call /query with
# an empty question to check auth. Simpler: peek at the tenant via a dry
# ingest attempt is messy. Best approach: we store tenant info from the first
# successful /query or /ingest response.
#
# For immediate feedback we do a lightweight OPTIONS-style check: call /health
# (public), then mark the key as "pending verification" until first real call.
# ---------------------------------------------------------------------------

def verify_key(api_key: str) -> dict | None:
    """
    Send a minimal /query request to verify the key.
    Returns tenant info dict on success, None on auth failure.
    """
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            headers={"X-API-Key": api_key},
            data={"question": "__ping__"},
            timeout=10,
        )
        if resp.status_code == 401:
            return None
        if resp.status_code in (200, 404):
            # 404 = collection not found yet (no PDFs ingested), key is still valid
            data = resp.json()
            return {
                "tenant": data.get("tenant", "unknown"),
                "collection": data.get("collection", "unknown"),
            }
        return None
    except requests.ConnectionError:
        st.error("Cannot reach the API server. Is `uvicorn src.api:app --reload` running?")
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


def call_query(api_key: str, question: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/query",
        headers={"X-API-Key": api_key},
        data={"question": question},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


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
        # Key changed — reset state
        st.session_state.api_key = key_input
        st.session_state.tenant_info = None
        st.session_state.messages = []

    if st.session_state.api_key and st.session_state.tenant_info is None:
        with st.spinner("Verifying key..."):
            info = verify_key(st.session_state.api_key)
        if info:
            st.session_state.tenant_info = info
        else:
            st.error("Invalid or revoked API key.")

    if st.session_state.tenant_info:
        info = st.session_state.tenant_info
        st.success(f"✅ Connected")
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
                    st.markdown(
                        f"- **{s['source']}** — Page {s['page']} "
                        f"*(relevance: {s['score']})*"
                    )

# Chat input
question = st.chat_input("Ask a question about your policy documents…")

if question:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Call API and stream the answer into the chat bubble
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents…"):
            try:
                result = call_query(st.session_state.api_key, question)
                answer = result["answer"]
                sources = result.get("sources", [])

                # Update tenant info in case it wasn't set yet
                if not st.session_state.tenant_info:
                    st.session_state.tenant_info = {
                        "tenant": result.get("tenant"),
                        "collection": result.get("collection"),
                    }

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for s in sources:
                            st.markdown(
                                f"- **{s['source']}** — Page {s['page']} "
                                f"*(relevance: {s['score']})*"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.HTTPError as e:
                detail = e.response.json().get("detail", str(e))
                err = f"⚠️ API error: {detail}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
            except requests.ConnectionError:
                err = "⚠️ Cannot reach the API. Is the server running?"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
