"""Aggregation logic for F_consistency (no LLM / NLI calls)."""

from __future__ import annotations

import pytest

from src.eval.f_consistency import combine


def test_combine_judge_only_when_nli_unavailable():
    score, validity, j, n = combine(judge_likert_score=4, nli_p_contradiction=None, nli_p_entailment=None)
    assert validity == "judge_only_no_nli"
    assert j == pytest.approx(0.75)
    assert n is None
    assert score == pytest.approx(0.75)


def test_combine_high_judge_high_entailment_takes_min():
    # likert=5 → judge_norm=1.0; p_e=0.85, p_n=0.10, p_c=0.05 → nli_signal = 0.85 + 0.05 - 0.05 = 0.85
    # score = min(1.0, 0.85) = 0.85
    score, validity, _, n = combine(
        judge_likert_score=5, nli_p_contradiction=0.05, nli_p_entailment=0.85, nli_p_neutral=0.10
    )
    assert validity == "full"
    assert n == pytest.approx(0.85)
    assert score == pytest.approx(0.85)


def test_combine_pure_neutral_does_not_inflate():
    """Pure-neutral (evasive) response must not score ~0.95.
    With entailment-aware combine: p_n=1.0, p_e=0.0, p_c=0.0 → nli_signal = 0.5."""
    score, _, _, n = combine(
        judge_likert_score=5, nli_p_contradiction=0.0, nli_p_entailment=0.0, nli_p_neutral=1.0
    )
    assert n == pytest.approx(0.5)
    assert score == pytest.approx(0.5)  # min(1.0, 0.5)


def test_combine_high_contradiction_collapses():
    # likert=5 → judge_norm=1.0; p_c=0.9, p_e=0.05, p_n=0.05 → nli_signal = 0.05 + 0.025 - 0.9 → clipped to 0
    score, _, _, n = combine(
        judge_likert_score=5, nli_p_contradiction=0.9, nli_p_entailment=0.05, nli_p_neutral=0.05
    )
    assert n == pytest.approx(0.0)
    assert score == pytest.approx(0.0)


def test_combine_low_judge_high_nli_takes_min():
    # likert=2 → judge_norm=0.25; high entailment → nli_signal high
    # score = min(0.25, ~0.85) = 0.25
    score, _, _, n = combine(
        judge_likert_score=2, nli_p_contradiction=0.05, nli_p_entailment=0.85, nli_p_neutral=0.10
    )
    assert score == pytest.approx(0.25)


def test_combine_clipping_extremes():
    # All combinations stay in [0, 1].
    s1, _, _, _ = combine(1, 1.0, 0.0, 0.0)
    s2, _, _, _ = combine(5, 0.0, 1.0, 0.0)
    assert 0.0 <= s1 <= 1.0
    assert 0.0 <= s2 <= 1.0


def test_combine_validity_full_when_nli_signals_present():
    _, validity, _, _ = combine(3, 0.1, 0.7, 0.2)
    assert validity == "full"


def test_combine_validity_judge_only_when_nli_missing():
    _, validity, _, _ = combine(3, None, None, None)
    assert validity == "judge_only_no_nli"
