"""Validate that the hand-curated corpus files load cleanly via the pydantic schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.schemas import FactKB, PersonaCorpus, Prompt, PromptCorpus

CORPUS = Path(__file__).resolve().parents[1] / "src" / "demos" / "vaxbot" / "corpus"


def _load(name: str) -> dict:
    return json.loads((CORPUS / name).read_text(encoding="utf-8"))


def test_prompts_load_and_shape():
    raw = _load("prompts.json")
    corpus = PromptCorpus.model_validate(raw)

    assert len(corpus.prompts) == 30, "expected 30 base prompts"

    by_cat: dict[str, list[Prompt]] = {}
    for p in corpus.prompts:
        by_cat.setdefault(p.category, []).append(p)

    assert len(by_cat["single_turn_factual"]) == 15
    assert len(by_cat["multi_turn"]) == 10
    assert len(by_cat["high_stakes"]) == 5

    # IDs are unique.
    ids = [p.id for p in corpus.prompts]
    assert len(set(ids)) == len(ids)

    # Multi-turn entries have exactly 3 turns + a consistency_probe.
    for p in by_cat["multi_turn"]:
        assert p.turns is not None and len(p.turns) == 3, p.id
        assert p.consistency_probe is not None, p.id
        assert 1 <= p.consistency_probe.anchor_turn <= 3
        assert 1 <= p.consistency_probe.probe_turn <= 3
        assert p.consistency_probe.probe_turn != p.consistency_probe.anchor_turn

    # Single-turn / high-stakes entries have a prompt string, no turns.
    for p in by_cat["single_turn_factual"] + by_cat["high_stakes"]:
        assert p.prompt is not None and len(p.prompt) > 0, p.id
        assert p.turns is None, p.id

    # High-stakes entries declare an expected_behavior.
    for p in by_cat["high_stakes"]:
        assert p.expected_behavior is not None, p.id


def test_personas_load_and_shape():
    raw = _load("personas.json")
    corpus = PersonaCorpus.model_validate(raw)

    assert corpus.axes == ["caste", "religion", "ses"]
    assert len(corpus.pairs) == 3

    pair_ids = {p.pair_id for p in corpus.pairs}
    assert pair_ids == {"caste", "religion", "ses"}

    for pair in corpus.pairs:
        # The varied attribute is named in held_constant's complement, in spirit.
        assert pair.varied_attribute, pair.pair_id
        assert pair.a.id != pair.b.id
        # Both arms have a non-empty injection preamble.
        assert len(pair.a.injection_preamble) > 50
        assert len(pair.b.injection_preamble) > 50


def test_facts_load_and_cover_referenced_ids():
    raw_facts = _load("facts.json")
    kb = FactKB.model_validate(raw_facts)
    assert len(kb.facts) >= 30, f"expected ~30 facts, got {len(kb.facts)}"

    fact_ids = set(kb.by_id().keys())

    # Every fact ID referenced by any prompt's expected_facts must exist in the KB.
    raw_prompts = _load("prompts.json")
    corpus = PromptCorpus.model_validate(raw_prompts)
    referenced: set[str] = set()
    for p in corpus.prompts:
        referenced.update(p.expected_facts)

    missing = referenced - fact_ids
    assert not missing, f"prompts reference undefined fact IDs: {sorted(missing)}"
