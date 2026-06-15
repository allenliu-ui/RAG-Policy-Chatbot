"""
pages/2_Documents.py — Upload and manage policy documents.
"""

import streamlit as st
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.ui_utils import require_auth, logout, call_ingest, call_list_documents

st.set_page_config(
    page_title="Documents — Policy Chatbot",
    page_icon="📄",
    layout="centered",
)

require_auth()

info = st.session_state.tenant_info

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📋 Policy Chatbot")
    st.caption(f"**{info['tenant']}** · `{info['collection']}`")
    st.markdown("---")
    st.page_link("pages/1_Chat.py", label="💬 Chat")
    st.page_link("pages/3_Admin.py", label="🔧 Admin Dashboard")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        logout()

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------

st.markdown("## 📄 Documents")
st.caption(f"Collection: `{info['collection']}`")
st.markdown("---")

st.markdown("### Upload a PDF")

uploaded = st.file_uploader(
    "Drag and drop a PDF here or click to browse",
    type=["pdf"],
    help="The PDF will be chunked, embedded, and stored in your collection.",
    label_visibility="collapsed",
)

if uploaded:
    st.markdown(f"**Selected:** {uploaded.name} ({uploaded.size / 1024:.1f} KB)")
    if st.button("⬆️ Upload & Ingest", type="primary", use_container_width=False):
        with st.spinner(f"Processing {uploaded.name}… this may take a moment."):
            try:
                result = call_ingest(st.session_state.api_key, uploaded)
                st.success(
                    f"✅ **{result['filename']}** ingested successfully — "
                    f"{result['chunks_stored']} chunks stored."
                )
                st.session_state.documents = call_list_documents(st.session_state.api_key)
                st.session_state.filter_sources = []
            except requests.HTTPError as e:
                try:
                    detail = e.response.json().get("detail", str(e))
                except Exception:
                    detail = str(e)
                st.error(f"❌ Upload failed: {detail}")
            except Exception as e:
                st.error(f"❌ {e}")

# ---------------------------------------------------------------------------
# Documents list
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Ingested Documents")

col1, col2 = st.columns([6, 1])
col1.caption(f"{len(st.session_state.documents)} document(s) in this collection.")
if col2.button("🔄 Refresh"):
    st.session_state.documents = call_list_documents(st.session_state.api_key)
    st.rerun()

if not st.session_state.documents:
    st.info("No documents yet. Upload a PDF above to get started.")
else:
    for doc in st.session_state.documents:
        st.markdown(f"📄 {doc}")
