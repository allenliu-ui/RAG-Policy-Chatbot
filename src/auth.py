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
from typing import Optional, List

KEYS_FILE = Path(__file__).parent.parent / "data" / "api_keys.json"
KEY_PREFIX = "rag_"
KEY_BYTES = 32


def _load() -> dict:
    if not KEYS_FILE.exists():
        return {}
    with open(KEYS_FILE) as f:
        return json.load(f)


def _save(keys: dict) -> None:
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)


def create_key(tenant: str, collection: str) -> str:
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


def validate_key(api_key: str) -> Optional[dict]:
    keys = _load()
    info = keys.get(api_key)
    if info and info.get("active", True):
        return info
    return None


def revoke_key(api_key: str) -> bool:
    keys = _load()
    if api_key not in keys:
        return False
    keys[api_key]["active"] = False
    _save(keys)
    return True


def revoke_by_hint(key_hint: str) -> bool:
    suffix = key_hint.split("...")[-1] if "..." in key_hint else key_hint[-8:]
    keys = _load()
    for k, v in keys.items():
        if k.endswith(suffix) and v.get("active", True):
            v["active"] = False
            _save(keys)
            return True
    return False


def list_keys() -> List[dict]:
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
