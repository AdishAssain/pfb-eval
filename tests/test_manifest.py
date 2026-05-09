"""Manifest module: deterministic hashes, structure, and schema_version surfacing."""

from __future__ import annotations

from pathlib import Path

from src.eval.manifest import compute_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "src" / "demos" / "vaxbot"


def test_manifest_structure():
    m = compute_manifest(DEMO_DIR)
    assert "system_prompt" in m
    assert "judge_prompts" in m
    assert "corpus" in m
    assert m["system_prompt"]["path"].endswith("system_prompt.md")
    assert isinstance(m["system_prompt"]["sha256"], str) and len(m["system_prompt"]["sha256"]) == 64


def test_manifest_judge_prompts_complete():
    m = compute_manifest(DEMO_DIR)
    expected = {
        "claim_extraction.md",
        "claim_verification.md",
        "consistency_likert.md",
        "recommendation_extraction.md",
        "recommendation_comparison.md",
    }
    assert expected.issubset(set(m["judge_prompts"].keys()))
    for name, sha in m["judge_prompts"].items():
        assert isinstance(sha, str) and len(sha) == 64, name


def test_manifest_corpus_includes_schema_version():
    m = compute_manifest(DEMO_DIR)
    for fname in ("prompts.json", "personas.json", "facts.json"):
        entry = m["corpus"][fname]
        assert isinstance(entry["sha256"], str) and len(entry["sha256"]) == 64
        # All three corpus files declare schema_version at the top.
        assert entry["schema_version"] is not None, fname


def test_manifest_deterministic():
    m1 = compute_manifest(DEMO_DIR)
    m2 = compute_manifest(DEMO_DIR)
    assert m1 == m2
