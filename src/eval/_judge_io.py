"""Shared helpers for parsing judge JSON output."""

from __future__ import annotations

import json


def safe_json_loads(text: str) -> dict:
    """Robustly extract a JSON object from a judge response.

    Handles:
      - markdown ```json fences
      - leading prose like "Here is the JSON requested:" before the JSON
      - trailing commentary after the JSON
    Raises ValueError with a short prefix of the offending text if no JSON object is found.
    """
    if not text:
        raise ValueError("judge returned empty text")

    s = text.strip()

    # Strip markdown fences first.
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
        if s.startswith("json\n"):
            s = s[5:]
        s = s.strip()

    # If the text still has prose before / after the JSON object, slice on first { and last }.
    if not (s.startswith("{") and s.endswith("}")):
        first = s.find("{")
        last = s.rfind("}")
        if first >= 0 and last > first:
            s = s[first : last + 1]

    try:
        return json.loads(s)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge returned non-JSON: {text[:200]!r}") from exc
