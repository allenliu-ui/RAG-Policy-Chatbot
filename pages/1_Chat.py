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
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### 📋 Policy Chatbot")
    st.caption(f"**{info['tenant']}** · `{info['collection']}`")
    st.markdown("---")

    # Document filter
    if st.session_state.documents:
        st.markdown("**Filter Documents**")
        selected = st.multiselect(
            "Search only in:",
            options=st.session_state.documents,
            default=[d for d in st.session_state.filter_sources if d in st.session_state.documents],
            placeholder="All documents",
            label_visibility="collapsed",
        )
        st.session_state.filter_sources = selected
        if selected:
            st.caption(f"Filtering to {len(selected)} of {len(st.session_state.documents)} doc(s).")
        st.markdown("---")

    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if col2.button("🚪 Logout", use_container_width=True):
        logout()

    st.markdown("---")
    st.page_link("pages/2_Documents.py", label="📄 Manage Documents")
    st.page_link("pages/3_Admin.py", label="🔧 Admin Dashboard")

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
