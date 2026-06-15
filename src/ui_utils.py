"""
ui_utils.py — Shared helpers for all Streamlit pages.
"""

import json
import requests
import streamlit as st
from typing import Optional, List

API_BASE = "http://localhost:8000"
MAX_HISTORY_TURNS = 5


# ---------------------------------------------------------------------------
# Session state bootstrap — call once at the top of every page
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "api_key": "",
        "tenant_info": None,
        "documents": [],
        "messages": [],
        "filter_sources": [],
        "ingest_status": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def require_auth():
    """Stop rendering and show a login prompt if the user isn't authenticated."""
    init_session()
    if not st.session_state.tenant_info:
        st.warning("You need to log in first.")
        st.page_link("streamlit_app.py", label="Go to Login →")
        st.stop()


def logout():
    for key in ["api_key", "tenant_info", "documents", "messages", "filter_sources", "ingest_status"]:
        st.session_state[key] = None if key == "tenant_info" else ([] if isinstance(st.session_state[key], list) else "")
    st.switch_page("streamlit_app.py")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def verify_key(api_key: str) -> Optional[dict]:
    try:
        resp = requests.get(
            f"{API_BASE}/me",
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        return resp.json() if resp.status_code == 200 else None
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


def stream_query(api_key: str, question: str, history: List[dict], filter_sources: Optional[List[str]] = None):
    """
    Returns (token_generator, sources_list).
    Consume the generator fully via st.write_stream before reading sources_list.
    """
    sources_out: List[dict] = []

    def _gen():
        payload = {"question": question, "history": history}
        if filter_sources:
            payload["filter_sources"] = filter_sources
        with requests.post(
            f"{API_BASE}/query/stream",
            headers={"X-API-Key": api_key},
            json=payload,
            stream=True,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw if isinstance(raw, str) else raw.decode()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event["type"] == "token":
                    yield event["content"]
                elif event["type"] == "sources":
                    sources_out.extend(event["content"])

    return _gen(), sources_out


def build_history(messages: List[dict]) -> List[dict]:
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages[-(MAX_HISTORY_TURNS * 2):]
    ]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def confidence_badge(score: float) -> str:
    if score >= 0.80:
        return "🟢"
    elif score >= 0.60:
        return "🟡"
    return "🔴"


def confidence_label(score: float) -> str:
    if score >= 0.80:
        return "High"
    elif score >= 0.60:
        return "Medium"
    return "Low"


def render_sources(sources: List[dict]):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            badge = confidence_badge(s["score"])
            label = confidence_label(s["score"])
            st.markdown(
                f"{badge} **{s['source']}** — Page {s['page']} &nbsp; "
                f"`{label} relevance ({s['score']})`"
            )
