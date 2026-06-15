"""
streamlit_app.py — Login page for the RAG Policy Chatbot.

This is the app entry point. Authenticated users are sent to the Chat page.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.ui_utils import init_session, verify_key, call_list_documents

st.set_page_config(
    page_title="Policy Chatbot — Login",
    page_icon="📋",
    layout="centered",
)

init_session()

# If already authenticated, go straight to chat
if st.session_state.tenant_info:
    st.switch_page("pages/1_Chat.py")

# ---------------------------------------------------------------------------
# Login UI
# ---------------------------------------------------------------------------

st.markdown("<br><br>", unsafe_allow_html=True)

col_l, col_mid, col_r = st.columns([1, 2, 1])
with col_mid:
    st.markdown("## 📋 Policy Chatbot")
    st.markdown("Your company's AI-powered policy assistant.")
    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("login_form"):
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="rag_...",
            help="Contact your administrator if you don't have a key.",
        )
        submitted = st.form_submit_button("Connect →", use_container_width=True, type="primary")

    if submitted:
        if not api_key:
            st.error("Please enter an API key.")
        else:
            with st.spinner("Verifying key…"):
                info = verify_key(api_key)
            if info:
                st.session_state.api_key = api_key
                st.session_state.tenant_info = info
                st.session_state.documents = call_list_documents(api_key)
                st.switch_page("pages/1_Chat.py")
            else:
                st.error("Invalid or revoked API key. Please try again.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Admin? Use the Admin page to create and manage API keys.")
