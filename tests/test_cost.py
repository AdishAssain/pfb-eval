"""Cost calculation + cumulative cost ceiling enforcement."""

from __future__ import annotations

import pytest

from src.eval import llm_client
from src.eval.llm_client import CostCeilingExceeded, _cost_usd


def test_cost_calc_known_pricing():
    # gpt-4o-mini: $0.15 in, $0.60 out per 1M tokens.
    # 1000 in + 500 out = 1000 * 0.15 / 1e6 + 500 * 0.60 / 1e6 = 0.00015 + 0.00030 = 0.00045
    cost = _cost_usd("openai/gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    assert cost == pytest.approx(0.00045, rel=1e-9)


def test_cost_calc_unknown_model_raises():
    """Fail closed: an unknown model id must raise, not silently bypass cost tracking."""
    from src.eval.llm_client import UnknownModelPricing

    with pytest.raises(UnknownModelPricing):
        _cost_usd("unknown/no-model", prompt_tokens=1000, completion_tokens=1000)


def test_cost_ceiling_raises(monkeypatch):
    monkeypatch.setenv("PFB_MAX_USD", "0.01")
    # Reset both module-level cost counters.
    monkeypatch.setattr(llm_client, "_cost_billed_usd", 0.0)
    monkeypatch.setattr(llm_client, "_cost_attributed_usd", 0.0)

    # First record under ceiling: ok.
    llm_client._record_cost(0.005)
    # Second record pushes billed total over ceiling: raises.
    with pytest.raises(CostCeilingExceeded):
        llm_client._record_cost(0.010)


def test_cost_ceiling_ignores_cached_calls(monkeypatch):
    """Cache hits attribute cost (for reporting) but must NOT trip the billed-cost ceiling."""
    monkeypatch.setenv("PFB_MAX_USD", "0.01")
    monkeypatch.setattr(llm_client, "_cost_billed_usd", 0.0)
    monkeypatch.setattr(llm_client, "_cost_attributed_usd", 0.0)

    # Many cached calls way over ceiling — no raise, just attributed cost.
    for _ in range(100):
        llm_client._record_cost(0.005, cached=True)
    assert llm_client.cost_billed_usd() == 0.0
    assert llm_client.cost_total_usd() == pytest.approx(0.5)
