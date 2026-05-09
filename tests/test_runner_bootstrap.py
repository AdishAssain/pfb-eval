"""Pure-function tests for the runner's bootstrap CI."""

from __future__ import annotations

import math

from src.eval.runner import bootstrap_ci


def test_bootstrap_ci_empty():
    r = bootstrap_ci([])
    assert r["n"] == 0
    assert math.isnan(r["mean"])


def test_bootstrap_ci_single_value():
    r = bootstrap_ci([0.5])
    assert r["n"] == 1
    assert r["mean"] == 0.5
    assert r["ci_lo"] == 0.5
    assert r["ci_hi"] == 0.5


def test_bootstrap_ci_known_values():
    values = [0.5, 0.6, 0.7, 0.8, 0.9]
    r = bootstrap_ci(values, n=500)
    # Mean should match the sample mean exactly.
    assert abs(r["mean"] - 0.7) < 1e-9
    # CI should bracket the mean.
    assert r["ci_lo"] <= r["mean"] <= r["ci_hi"]
    # CI should be within the data range.
    assert r["ci_lo"] >= min(values) - 0.05
    assert r["ci_hi"] <= max(values) + 0.05


def test_bootstrap_ci_seed_deterministic():
    r1 = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n=200, seed=42)
    r2 = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n=200, seed=42)
    assert r1 == r2
