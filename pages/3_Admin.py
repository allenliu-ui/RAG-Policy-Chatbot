"""
pages/3_Admin.py — Admin dashboard for the RAG Policy Chatbot.

Accessible at http://localhost:8501/3_Admin when the Streamlit app is running.
Requires the FastAPI backend (uvicorn src.api:app --reload) to be running.
"""

import streamlit as st
import requests
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Admin — Policy Chatbot", page_icon="🔧", layout="wide")

with st.sidebar:
    st.markdown("### 📋 Policy Chatbot")
    st.markdown("---")
    st.page_link("streamlit_app.py", label="← Back to Login")
    st.page_link("pages/1_Chat.py", label="💬 Chat")
    st.page_link("pages/2_Documents.py", label="📄 Documents")

st.title("🔧 Admin Dashboard")
st.caption("Manage tenant keys, view collection stats, and revoke access.")

# ---------------------------------------------------------------------------
# Admin key gate
# ---------------------------------------------------------------------------

admin_key = st.text_input("Admin Key", type="password", placeholder="Enter ADMIN_SECRET")

if not admin_key:
    st.info("Enter your admin key above to get started.")
    st.stop()

headers = {"X-Admin-Key": admin_key}

try:
    keys_resp = requests.get(f"{API_BASE}/admin/keys", headers=headers, timeout=10)
    if keys_resp.status_code in (401, 403):
        st.error("Invalid admin key.")
        st.stop()
    keys_resp.raise_for_status()
    keys = keys_resp.json()
except requests.ConnectionError:
    st.error("Cannot reach the API server. Is it running?")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error: {e}")
    st.stop()

try:
    stats_resp = requests.get(f"{API_BASE}/admin/stats", headers=headers, timeout=15)
    stats_by_collection = (
        {s["collection"]: s for s in stats_resp.json()} if stats_resp.ok else {}
    )
except Exception:
    stats_by_collection = {}

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

active_count = sum(1 for k in keys if k["active"])
collections = {k["collection"] for k in keys}

col1, col2, col3 = st.columns(3)
col1.metric("Total Keys", len(keys))
col2.metric("Active Keys", active_count)
col3.metric("Collections", len(collections))

st.markdown("---")

# ---------------------------------------------------------------------------
# Create new tenant key
# ---------------------------------------------------------------------------

with st.expander("➕ Create New Tenant Key"):
    with st.form("create_key_form"):
        new_tenant = st.text_input("Tenant name", placeholder="acme_corp")
        new_collection = st.text_input("Collection name", placeholder="acme_policies")
        submitted = st.form_submit_button("Create Key", type="primary")
        if submitted:
            if not new_tenant or not new_collection:
                st.error("Both fields are required.")
            else:
                try:
                    r = requests.post(
                        f"{API_BASE}/admin/keys",
                        headers={**headers, "Content-Type": "application/json"},
                        json={"tenant": new_tenant, "collection": new_collection},
                        timeout=10,
                    )
                    r.raise_for_status()
                    data = r.json()
                    st.success(f"Key created for **{new_tenant}**. Copy it now — it won't be shown again.")
                    st.code(data["api_key"])
                except Exception as e:
                    st.error(f"Failed to create key: {e}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tenant key table
# ---------------------------------------------------------------------------

st.subheader(f"Tenant Keys ({len(keys)})")

if not keys:
    st.info("No tenant keys found.")
else:
    h = st.columns([2, 2, 2, 1, 1, 1, 1])
    for col, label in zip(h, ["Tenant", "Collection", "Key hint", "Created", "Chunks", "Docs", "Action"]):
        col.markdown(f"**{label}**")

    st.divider()

    for key_info in keys:
        coll = key_info["collection"]
        stats = stats_by_collection.get(coll, {})
        row = st.columns([2, 2, 2, 1, 1, 1, 1])

        row[0].write(key_info["tenant"])
        row[1].write(coll)
        row[2].code(key_info["key_hint"], language=None)
        row[3].write(key_info["created_at"][:10])
        row[4].write(str(stats.get("point_count", "—")))
        row[5].write(str(stats.get("document_count", "—")))

        if key_info["active"]:
            if row[6].button("Revoke", key=f"revoke_{key_info['key_hint']}"):
                try:
                    r = requests.post(
                        f"{API_BASE}/admin/keys/revoke-by-hint",
                        headers={**headers, "Content-Type": "application/json"},
                        json={"key_hint": key_info["key_hint"]},
                        timeout=10,
                    )
                    r.raise_for_status()
                    st.success(f"Revoked key for **{key_info['tenant']}**.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to revoke: {e}")
        else:
            row[6].markdown("❌ Revoked")
