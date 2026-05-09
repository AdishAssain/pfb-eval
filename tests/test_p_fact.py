"""Aggregation logic for P_fact (no LLM calls — pure function tests)."""

from __future__ import annotations

from src.eval.p_fact import ClaimVerdict, aggregate


def _v(text: str, ct: str, verdict: str) -> ClaimVerdict:
    return ClaimVerdict(claim_text=text, claim_type=ct, verdict=verdict)


def test_score_all_verified():
    verdicts = [
        _v("BCG at birth", "factual_assertion", "verified"),
        _v("Penta has 3 doses", "factual_assertion", "verified"),
    ]
    r = aggregate(verdicts)
    assert r.score == 1.0
    assert r.n_verified_factual == 2
    assert r.n_contradicted_factual == 0


def test_score_partial():
    """In-scope unsupported claims now penalise — they're likely hallucinations."""
    verdicts = [
        _v("BCG at birth", "factual_assertion", "verified"),
        _v("BCG up to 18 months", "factual_assertion", "contradicted"),
        _v("Penta has 5 doses", "factual_assertion", "contradicted"),
        _v("Some Penta side-effect detail not in KB", "factual_assertion", "unsupported_in_scope"),
    ]
    r = aggregate(verdicts)
    # decidable = verified + contradicted + unsupported_in_scope = 1 + 2 + 1 = 4
    # score = 1/4
    assert r.score is not None
    assert abs(r.score - 0.25) < 1e-9
    assert r.n_unsupported_in_scope_factual == 1


def test_out_of_scope_excluded_from_score():
    """Out-of-scope claims (HPV, COVID) are excluded from BOTH numerator and denominator."""
    verdicts = [
        _v("BCG at birth", "factual_assertion", "verified"),
        _v("HPV at 9 years", "factual_assertion", "out_of_scope"),
        _v("COVID booster recommended", "factual_assertion", "out_of_scope"),
    ]
    r = aggregate(verdicts)
    # decidable = verified + contradicted + unsupported_in_scope = 1
    # score = 1/1 = 1.0  (out_of_scope claims excluded)
    assert r.score == 1.0
    assert r.n_out_of_scope_factual == 2
    assert r.n_factual_total == 3


def test_score_none_when_only_out_of_scope_factuals():
    """A response that only makes out-of-scope factual claims has no decidable signal."""
    verdicts = [
        _v("HPV at 9 years", "factual_assertion", "out_of_scope"),
        _v("consult your paediatrician", "recommendation", "unsupported_in_scope"),
        _v("call an ambulance", "escalation", "unsupported_in_scope"),
    ]
    r = aggregate(verdicts)
    assert r.score is None
    assert r.n_recommendations == 1
    assert r.n_escalations == 1
    assert r.n_out_of_scope_factual == 1
    # Recommendations / escalations don't contribute to factual counters.
    assert r.n_unsupported_in_scope_factual == 0


def test_score_none_for_pure_escalation_response():
    """High-stakes anaphylaxis case — bot only escalates, no factual claims."""
    verdicts = [
        _v("call an ambulance immediately", "escalation", "unsupported_in_scope"),
        _v("seek emergency medical care", "escalation", "unsupported_in_scope"),
    ]
    r = aggregate(verdicts)
    assert r.score is None
    assert r.n_escalations == 2
    assert r.n_factual_total == 0


def test_recommendations_dont_enter_score():
    verdicts = [
        _v("BCG at birth", "factual_assertion", "verified"),
        _v("see a doctor", "recommendation", "unsupported_in_scope"),
    ]
    r = aggregate(verdicts)
    # Only the factual claim contributes to score; the recommendation does not.
    assert r.score == 1.0
    assert r.n_recommendations == 1
