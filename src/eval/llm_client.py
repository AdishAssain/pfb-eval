"""Async-first OpenRouter chat wrapper used by the harness and every demo.

Every model in the project routes through OpenRouter's OpenAI-compatible API,
so a single async client handles GPT, Gemini, Claude, etc. via provider-prefixed
model names (`openai/gpt-4o-mini`, `google/gemini-2.5-flash`, ...).

Features:
  - Async-first (`chat_async`) with a sync wrapper (`chat`) for CLI / scripts.
  - Disk cache keyed on model + canonical messages + params; bypass with PFB_NO_CACHE=1.
  - Retry-with-jitter on transient errors.
  - Per-call timeout (default 60s; override with PFB_TIMEOUT_S).
  - Cumulative cost ceiling across the process (default USD 25; override with PFB_MAX_USD).
"""

from __future__ import annotations

import asyncio
import os
from threading import Lock

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.eval import cache, log as logmod
from src.eval.schemas import ChatResponse


# Errors we retry on. 4xx client errors (auth, permission, bad request, model-not-found)
# fail fast; retrying them just wastes time and money. 5xx server errors, network /
# timeout failures, and 429 rate limits are transient and worth retrying.
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,    # network blip / DNS failure
    APITimeoutError,       # request timed out
    RateLimitError,        # 429 — backoff and try again
    InternalServerError,   # 5xx — provider hiccup
)

load_dotenv()
_log = logmod.get("pfb.llm")


# USD per 1M tokens (input, output) via OpenRouter, as of run-date 2026-05-09.
# Refresh if a model row changes on openrouter.ai/models.
PRICING_USD_PER_M: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini":          (0.15,  0.60),
    "google/gemini-2.5-flash":     (0.30,  2.50),
    "anthropic/claude-haiku-4.5":  (1.00,  5.00),
    "google/gemini-2.5-pro":       (1.25, 10.00),
}


class CostCeilingExceeded(RuntimeError):
    pass


class OpenRouterCreditsExhausted(RuntimeError):
    """OpenRouter returned 402 / 'insufficient credits' / 'insufficient balance'.

    Fail fast (don't retry) and surface a friendly hint to the user via main.py.
    """
    pass


class UnknownModelPricing(RuntimeError):
    """Raised when a model is invoked whose USD pricing is not in PRICING_USD_PER_M.

    Fails closed by design: silently treating an unknown model as free would defeat
    the cumulative cost ceiling and hide budget overruns. Add the model to
    PRICING_USD_PER_M (with the up-to-date OpenRouter pricing) to fix.
    """
    pass


# Process-wide cost trackers (two separate counters).
#
#   _cost_billed_usd   — what we actually pay the provider this process. Cache hits do
#                        NOT contribute. This is the ceiling-enforcement counter.
#   _cost_attributed_usd — what the full evaluation would cost from scratch (no cache).
#                        Cache hits DO contribute (using the cached call's recorded cost).
#                        This is the value to surface in the public run summary so a
#                        re-run from cache does not appear to cost $0.
_cost_lock = Lock()
_cost_billed_usd: float = 0.0
_cost_attributed_usd: float = 0.0


def cost_total_usd() -> float:
    """Cost of this run as if no cache were available — surface this in run summaries."""
    return _cost_attributed_usd


def cost_billed_usd() -> float:
    """Actual amount paid to the provider in this process. Used for ceiling enforcement."""
    return _cost_billed_usd


def _max_usd() -> float:
    return float(os.environ.get("PFB_MAX_USD", "25.0"))


def _timeout_s() -> float:
    return float(os.environ.get("PFB_TIMEOUT_S", "60.0"))


_async_client: AsyncOpenAI | None = None


def _client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill it."
            )
        _async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            timeout=_timeout_s(),
            default_headers={
                "HTTP-Referer": "https://github.com/local/pfb-eval",
                "X-Title": "pfb-eval",
            },
        )
    return _async_client


def _cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if model not in PRICING_USD_PER_M:
        raise UnknownModelPricing(
            f"no pricing for {model!r}. add it to PRICING_USD_PER_M before calling."
        )
    in_per_m, out_per_m = PRICING_USD_PER_M[model]
    return (prompt_tokens * in_per_m + completion_tokens * out_per_m) / 1_000_000


def _record_cost(usd: float, *, cached: bool = False) -> None:
    global _cost_billed_usd, _cost_attributed_usd
    with _cost_lock:
        _cost_attributed_usd += usd
        if cached:
            return
        _cost_billed_usd += usd
        if _cost_billed_usd > _max_usd():
            raise CostCeilingExceeded(
                f"billed cost ${_cost_billed_usd:.4f} exceeded ceiling ${_max_usd():.2f}"
            )


async def chat_async(
    messages: list[dict[str, str]],
    model: str,
    *,
    temperature: float = 0.3,
    top_p: float = 0.9,
    max_tokens: int = 1024,
    max_retries: int = 4,
    response_format: dict | None = None,
    use_cache: bool = True,
) -> ChatResponse:
    params = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    if response_format is not None:
        params["response_format"] = response_format

    logmod.new_trace_id()

    if use_cache:
        ck = cache.cache_key(model, messages, **params)
        hit = cache.get(ck)
        if hit is not None:
            hit["cached"] = True
            cached_resp = ChatResponse.model_validate(hit)
            # Attribute the cached cost so the run summary reflects true methodology
            # cost; does NOT count against the billed-cost ceiling.
            _record_cost(cached_resp.usd_cost, cached=True)
            _log.info(
                "chat.cache_hit",
                extra={"event": {
                    "model": model, "key": ck[:12],
                    "attributed_usd": round(cached_resp.usd_cost, 6),
                }},
            )
            return cached_resp

    client = _client()
    last_err: Exception | None = None
    _log.info(
        "chat.start",
        extra={"event": {"model": model, "messages_n": len(messages), "max_tokens": max_tokens}},
    )

    # tenacity drives the retry-with-backoff. We retry only on transient errors
    # (network, timeout, rate-limit, 5xx); 4xx client errors (auth, bad request,
    # model-not-found) and CostCeilingExceeded fail fast.
    retryer = AsyncRetrying(
        stop=stop_after_attempt(max_retries),
        wait=wait_random_exponential(multiplier=1, max=30),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )

    attempt_idx = 0
    try:
        async for attempt in retryer:
            with attempt:
                attempt_idx = attempt.retry_state.attempt_number - 1
                try:
                    t0 = asyncio.get_event_loop().time()
                    r = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **params,
                    )
                    dt = asyncio.get_event_loop().time() - t0
                except CostCeilingExceeded:
                    raise
                except Exception as exc:
                    _log.warning(
                        "chat.retry",
                        extra={"event": {"model": model, "attempt": attempt_idx, "error": str(exc)[:200]}},
                    )
                    raise

                choice = r.choices[0]
                usage = r.usage
                ptok = getattr(usage, "prompt_tokens", 0) or 0
                ctok = getattr(usage, "completion_tokens", 0) or 0
                usd = _cost_usd(model, ptok, ctok)
                _record_cost(usd)

                resp = ChatResponse(
                    text=choice.message.content or "",
                    model=model,
                    prompt_tokens=ptok,
                    completion_tokens=ctok,
                    latency_seconds=dt,
                    usd_cost=usd,
                    finish_reason=choice.finish_reason or "",
                    cached=False,
                )

                if use_cache:
                    cache.put(ck, resp.model_dump())

                _log.info(
                    "chat.success",
                    extra={"event": {
                        "model": model, "ptok": ptok, "ctok": ctok,
                        "usd": round(usd, 6), "latency_s": round(dt, 3),
                        "attempt": attempt_idx, "finish": resp.finish_reason,
                        "running_billed_usd": round(_cost_billed_usd, 4),
                        "running_attributed_usd": round(_cost_attributed_usd, 4),
                    }},
                )
                return resp
    except CostCeilingExceeded:
        _log.error(
            "chat.cost_ceiling_exceeded",
            extra={"event": {
                "model": model,
                "running_billed_usd": round(_cost_billed_usd, 4),
                "running_attributed_usd": round(_cost_attributed_usd, 4),
            }},
        )
        raise
    except (UnknownModelPricing, OpenRouterCreditsExhausted):
        # Typed run-wide errors — propagate so main.py can render a friendly hint
        # rather than wrapping in the generic "chat failed after N attempts" runtime
        # error. UnknownModelPricing is raised by _cost_usd before any provider call;
        # OpenRouterCreditsExhausted is translated below from the upstream error.
        raise
    except RetryError as re:
        last_err = re.last_attempt.exception() if re.last_attempt else re
        _log.error(
            "chat.failed",
            extra={"event": {"model": model, "max_retries": max_retries, "error": str(last_err)[:300]}},
        )
        raise RuntimeError(f"chat failed after {max_retries} attempts for {model}") from last_err
    except Exception as exc:
        # 402 / credit-exhausted — translate to a typed error so main.py can render
        # a friendly hint instead of a stack trace.
        msg = str(exc).lower()
        if any(s in msg for s in ("insufficient credit", "insufficient balance", "402", "out of credits", "no credits")):
            _log.error("chat.credits_exhausted", extra={"event": {"model": model, "error": str(exc)[:200]}})
            raise OpenRouterCreditsExhausted(str(exc)) from exc
        _log.error(
            "chat.failed",
            extra={"event": {"model": model, "max_retries": max_retries, "error": str(exc)[:300]}},
        )
        raise RuntimeError(f"chat failed after {max_retries} attempts for {model}") from exc

    # Defensive: AsyncRetrying with reraise=True never reaches here, but keep mypy happy.
    raise RuntimeError(f"chat retry loop exited without returning for {model}")


def chat(messages: list[dict[str, str]], model: str, **kwargs) -> ChatResponse:
    """Sync wrapper around chat_async — do not call from inside async code (it spins a fresh loop)."""
    return asyncio.run(chat_async(messages, model, **kwargs))
