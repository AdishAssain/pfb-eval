"""On-disk JSON cache keyed by sha256 of model + canonical messages + params.

Bypass with env var PFB_NO_CACHE=1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "cache"

_log = logging.getLogger("pfb.cache")


def _enabled() -> bool:
    return os.environ.get("PFB_NO_CACHE", "0") not in ("1", "true", "True")


def cache_key(model: str, messages: list[dict[str, str]], **params) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "params": {k: params[k] for k in sorted(params)},
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get(key: str) -> dict[str, Any] | None:
    if not _enabled():
        return None
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        # A corrupt cache file (interrupted write, manual edit) must not be silently
        # surfaced as "chat failed after N attempts" upstream. Surface it loudly with
        # a path so the user can either delete the file or set PFB_NO_CACHE=1.
        _log.warning(
            "cache.corrupt",
            extra={"event": {
                "key": key[:12],
                "path": str(path),
                "error": str(exc)[:200],
                "action": "treating_as_cache_miss",
            }},
        )
        return None


def put(key: str, value: dict[str, Any]) -> None:
    if not _enabled():
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Atomic write: serialise to a sibling temp file then rename, so a process
    # killed mid-write cannot leave a half-written JSON that the next read would
    # silently swallow as a cache miss (or, worse, parse as malformed JSON).
    final = CACHE_DIR / f"{key}.json"
    tmp = CACHE_DIR / f"{key}.json.tmp.{os.getpid()}"
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, final)
