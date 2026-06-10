"""
auth.py — API key management for multi-tenant access

Each API key maps to a tenant name and a Qdrant collection.
Keys are stored in data/api_keys.json (gitignored).

Usage:
    from src.auth import create_key, validate_key, revoke_key, list_keys
"""

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

# Keys file sits at project root / data / api_keys.json
KEYS_FILE = Path(__file__).parent.parent / "data" / "api_keys.json"
KEY_PREFIX = "rag_"
KEY_BYTES = 32  # 256-bit entropy → 64-char hex string after prefix


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load() -> dict:
    if not KEYS_FILE.exists():
        return {}
    with open(KEYS_FILE) as f:
        return json.load(f)


def _save(keys: dict) -> None:
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_key(tenant: str, collection: str) -> str:
    """
    Generate a new API key for (tenant, collection) and persist it.

    Args:
        tenant:     Human-readable tenant name, e.g. "acme_corp".
        collection: Qdrant collection this key is scoped to.

    Returns:
        The new API key string (only shown once — store it safely).
    """
    api_key = KEY_PREFIX + secrets.token_hex(KEY_BYTES)
    keys = _load()
    keys[api_key] = {
        "tenant": tenant,
        "collection": collection,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": True,
    }
    _save(keys)
    return api_key


def validate_key(api_key: str) -> dict | None:
    """
    Return the tenant info dict if the key exists and is active, else None.

    Return shape: {"tenant": str, "collection": str, "created_at": str, "active": bool}
    """
    keys = _load()
    info = keys.get(api_key)
    if info and info.get("active", True):
        return info
    return None


def revoke_key(api_key: str) -> bool:
    """
    Mark a key inactive (soft-delete). Returns True if the key was found.
    """
    keys = _load()
    if api_key not in keys:
        return False
    keys[api_key]["active"] = False
    _save(keys)
    return True


def list_keys() -> list[dict]:
    """
    Return metadata for all keys. The key itself is redacted to its last 8
    characters so it can be displayed in admin UIs without full exposure.
    """
    keys = _load()
    return [
        {
            "key_hint": f"rag_...{k[-8:]}",
            "tenant": v["tenant"],
            "collection": v["collection"],
            "created_at": v["created_at"],
            "active": v["active"],
        }
        for k, v in keys.items()
    ]
