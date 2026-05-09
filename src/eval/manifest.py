"""Run manifest: SHA-256 hashes of every input artefact, embedded in run metadata.

This pins a result to the exact prompts, corpus, and knowledge base that produced
it. Reproducibility check: re-running with the same manifest hashes should yield
the same scores up to LLM non-determinism.

Tracked artefacts:
  - The chatbot system prompt under test (`src/demos/<demo>/system_prompt.md`).
  - Every judge prompt under `src/eval/judge_prompts/`.
  - Each corpus file (prompts, personas, facts) under the active demo's corpus dir.

We hash file contents only — not the runtime model identifiers, which travel
in `panel_models` and the `*_judge_model` fields of each per-prompt result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JUDGE_PROMPTS_DIR = Path(__file__).parent / "judge_prompts"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _maybe_schema_version(p: Path) -> str | None:
    """Pull a top-level `schema_version` from a JSON file, if present. Otherwise None.

    Narrow except: we only swallow JSON-format / IO errors. A malformed corpus file
    will fail loudly later in `_load_corpus`; here we just skip the schema_version
    field rather than aborting manifest construction.
    """
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(obj, dict):
        v = obj.get("schema_version")
        if isinstance(v, str):
            return v
    return None


def compute_manifest(demo_dir: Path) -> dict:
    """Build a manifest dict for a run that uses the given demo directory.

    Fails loudly if foundational artefacts (system_prompt.md, judge_prompts dir) are
    missing — silently writing null to the manifest would let an unreproducible run
    look reproducible.
    """
    system_prompt = demo_dir / "system_prompt.md"
    corpus_dir = demo_dir / "corpus"

    if not system_prompt.exists():
        raise FileNotFoundError(
            f"system_prompt.md missing at {system_prompt} — refusing to write a manifest "
            "with system_prompt.sha256=null (would silently break reproducibility)."
        )
    if not JUDGE_PROMPTS_DIR.exists():
        raise FileNotFoundError(
            f"judge_prompts dir missing at {JUDGE_PROMPTS_DIR} — refusing to write a manifest "
            "with empty judge_prompts (would silently break reproducibility)."
        )

    judge_prompts: dict[str, str] = {}
    for jp in sorted(JUDGE_PROMPTS_DIR.glob("*.md")):
        judge_prompts[jp.name] = _sha256(jp)
    if not judge_prompts:
        raise FileNotFoundError(
            f"no judge prompt files found under {JUDGE_PROMPTS_DIR} — refusing to write a "
            "manifest with empty judge_prompts (would silently break reproducibility)."
        )

    corpus: dict[str, dict[str, str | None]] = {}
    if corpus_dir.exists():
        for cf in sorted(corpus_dir.glob("*.json")):
            corpus[cf.name] = {
                "sha256": _sha256(cf),
                "schema_version": _maybe_schema_version(cf),
            }

    return {
        "system_prompt": {
            "path": str(system_prompt.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(system_prompt),
        },
        "judge_prompts": judge_prompts,
        "corpus": corpus,
    }
