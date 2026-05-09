"""Cache key determinism + put/get round-trip + PFB_NO_CACHE bypass."""

from __future__ import annotations

import os

from src.eval import cache


def test_cache_key_is_deterministic():
    msgs = [{"role": "user", "content": "hello"}]
    a = cache.cache_key("openai/gpt-4o-mini", msgs, temperature=0.3, max_tokens=128)
    b = cache.cache_key("openai/gpt-4o-mini", msgs, temperature=0.3, max_tokens=128)
    assert a == b


def test_cache_key_param_order_independent():
    msgs = [{"role": "user", "content": "hi"}]
    a = cache.cache_key("openai/gpt-4o-mini", msgs, temperature=0.3, max_tokens=128, top_p=0.9)
    b = cache.cache_key("openai/gpt-4o-mini", msgs, max_tokens=128, top_p=0.9, temperature=0.3)
    assert a == b


def test_cache_key_differs_with_inputs():
    base = cache.cache_key("openai/gpt-4o-mini", [{"role": "user", "content": "x"}], temperature=0.3)
    diff_model = cache.cache_key("google/gemini-2.5-flash", [{"role": "user", "content": "x"}], temperature=0.3)
    diff_msg = cache.cache_key("openai/gpt-4o-mini", [{"role": "user", "content": "y"}], temperature=0.3)
    diff_param = cache.cache_key("openai/gpt-4o-mini", [{"role": "user", "content": "x"}], temperature=0.5)
    assert len({base, diff_model, diff_msg, diff_param}) == 4


def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("PFB_NO_CACHE", raising=False)

    payload = {"text": "hello", "model": "x", "prompt_tokens": 1, "completion_tokens": 1,
               "latency_seconds": 0.1, "usd_cost": 0.0, "finish_reason": "stop", "cached": False}
    cache.put("abc123", payload)

    got = cache.get("abc123")
    assert got == payload


def test_cache_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("PFB_NO_CACHE", "1")

    cache.put("abc123", {"foo": "bar"})
    # When PFB_NO_CACHE=1, get() must return None even if a file would be present.
    assert cache.get("abc123") is None
    # And put() must NOT have written anything.
    assert not (tmp_path / "abc123.json").exists()
