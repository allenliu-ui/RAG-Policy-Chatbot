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

st.markdown("""
<style>
.sidebar-brand {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 8px;
    color: white;
}
.sidebar-brand h2 { margin: 0 0 2px 0; font-size: 1.15rem; font-weight: 700; color: white !important; }
.sidebar-brand p  { margin: 0; font-size: 0.75rem; opacity: 0.8; color: white !important; }
.tenant-card {
    background: #f0f4f8;
    border-left: 3px solid #2d6a9f;
    border-radius: 6px;
    padding: 10px 12px;
    margin: 8px 0;
}
.tenant-card .label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em; color: #6b7280; margin-bottom: 2px; }
.tenant-card .value { font-size: 0.85rem; font-weight: 600; color: #1e3a5f; }
.section-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; font-weight: 600; margin: 16px 0 6px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>📋 Policy Chatbot</h2>
        <p>AI-powered policy assistant</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="tenant-card">
        <div class="label">Tenant</div>
        <div class="value">{info['tenant']}</div>
        <div class="label" style="margin-top:6px;">Collection</div>
        <div class="value" style="font-size:0.78rem;">{info['collection']}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
    st.page_link("pages/1_Chat.py", label="💬 Chat", use_container_width=True)
    st.page_link("pages/3_Admin.py", label="🔧 Admin Dashboard", use_container_width=True)
    st.markdown('<div class="section-label">Actions</div>', unsafe_allow_html=True)
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
