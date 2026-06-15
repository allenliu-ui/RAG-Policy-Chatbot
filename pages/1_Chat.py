"""
pages/1_Chat.py — Main chat interface for the RAG Policy Chatbot.
"""

import streamlit as st
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.ui_utils import (
    require_auth, logout, build_history, stream_query,
    render_sources, call_list_documents,
)

st.set_page_config(
    page_title="Chat — Policy Chatbot",
    page_icon="💬",
    layout="wide",
)

require_auth()

info = st.session_state.tenant_info

# ---------------------------------------------------------------------------
# Sidebar styles
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Brand header block */
.sidebar-brand {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 8px;
    color: white;
}
.sidebar-brand h2 {
    margin: 0 0 2px 0;
    font-size: 1.15rem;
    font-weight: 700;
    color: white !important;
}
.sidebar-brand p {
    margin: 0;
    font-size: 0.75rem;
    opacity: 0.8;
    color: white !important;
}

/* Tenant info card */
.tenant-card {
    background: #f0f4f8;
    border-left: 3px solid #2d6a9f;
    border-radius: 6px;
    padding: 10px 12px;
    margin: 8px 0;
}
.tenant-card .label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280;
    margin-bottom: 2px;
}
.tenant-card .value {
    font-size: 0.85rem;
    font-weight: 600;
    color: #1e3a5f;
}

/* Section label */
.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    font-weight: 600;
    margin: 16px 0 6px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Brand header
    st.markdown(f"""
    <div class="sidebar-brand">
        <h2>📋 Policy Chatbot</h2>
        <p>AI-powered policy assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Tenant info card
    st.markdown(f"""
    <div class="tenant-card">
        <div class="label">Tenant</div>
        <div class="value">{info['tenant']}</div>
        <div class="label" style="margin-top:6px;">Collection</div>
        <div class="value" style="font-size:0.78rem;">{info['collection']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Document filter
    if st.session_state.documents:
        st.markdown('<div class="section-label">Filter by document</div>', unsafe_allow_html=True)
        selected = st.multiselect(
            "Filter",
            options=st.session_state.documents,
            default=[d for d in st.session_state.filter_sources if d in st.session_state.documents],
            placeholder="All documents",
            label_visibility="collapsed",
        )
        st.session_state.filter_sources = selected
        if selected:
            st.caption(f"Searching {len(selected)} of {len(st.session_state.documents)} doc(s).")

    # Navigation
    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    st.page_link("pages/2_Documents.py", label="📄 Manage Documents", use_container_width=True)
    st.page_link("pages/3_Admin.py", label="🔧 Admin Dashboard", use_container_width=True)

    # Actions
    st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col2.button("🚪 Logout", use_container_width=True):
        logout()

# ---------------------------------------------------------------------------
# Chat area
# ---------------------------------------------------------------------------

st.markdown("### Ask a policy question")

if not st.session_state.documents:
    st.info("No documents uploaded yet. Go to **Manage Documents** to upload your first PDF.")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# Input
question = st.chat_input("Ask a question about your policy documents…")

if question:
    history = build_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            token_gen, sources_ref = stream_query(
                st.session_state.api_key,
                question,
                history,
                filter_sources=st.session_state.filter_sources or None,
            )
            answer = st.write_stream(token_gen)
            render_sources(sources_ref)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_ref,
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
