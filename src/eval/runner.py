"""Eval runner: orchestrates the full pipeline across prompts × models × axes.

Scope:
  * Panel models: openai/gpt-4o-mini, google/gemini-2.5-flash (both via OpenRouter).
  * Corpus: 30 prompts (15 single-turn factual + 10 multi-turn × 3 turns + 5 high-stakes).
  * Persona axes: 3 paired (caste, religion, ses).
  * Bias runs on the 20 single-turn prompts only; multi-turn bias is out of scope.
  * F_consistency runs on the 10 multi-turn prompts.
  * P_fact runs on every base / final-turn response.
  * Naive (non-clustered) bootstrap 95% CIs, n=1000 resamples.

Output: results/run-<ts>/{results-public.json, results-private.json, run-summary.json}.
Public JSON omits demographic-tagged free-text; private JSON retains everything for
local reproduction.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.demos.vaxbot.client import system_prompt_sha256, vaxbot_chat_async
from src.eval import log as logmod
from src.eval.bias import (
    BiasAxisAggregate,
    BiasPairResult,
    aggregate_axis,
    bias_pair,
)
from src.eval.f_consistency import FConsistencyResult, f_consistency
from src.eval.llm_client import (
    CostCeilingExceeded,
    OpenRouterCreditsExhausted,
    UnknownModelPricing,
    cost_billed_usd,
    cost_total_usd,
)
from src.eval.manifest import compute_manifest
from src.eval.p_fact import PFactResult, p_fact
from src.eval.schemas import (
    FactKB,
    Persona,
    PersonaCorpus,
    PersonaPair,
    Prompt,
    PromptCorpus,
)


PANEL_MODELS = ["openai/gpt-4o-mini", "google/gemini-2.5-flash"]

CORPUS_DIR = Path(__file__).resolve().parents[1] / "demos" / "vaxbot" / "corpus"
PROMPTS_PATH = CORPUS_DIR / "prompts.json"
PERSONAS_PATH = CORPUS_DIR / "personas.json"
FACTS_PATH = CORPUS_DIR / "facts.json"

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
ASYNC_CONCURRENCY = 8  # bot/judge concurrent in-flight; respects OpenRouter rate limits


class TurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    turn_index: int  # 1-based
    user_text: str
    bot_text: str
    latency_seconds: float
    tokens_in: int
    tokens_out: int
    usd_cost: float
    cached: bool


class PerModelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    base_response: TurnResponse
    multi_turn_responses: list[TurnResponse] = Field(default_factory=list)
    p_fact: PFactResult | None = None
    f_consistency: FConsistencyResult | None = None
    bias_results: list[BiasPairResult] = Field(default_factory=list)
    safety_signal_present: bool | None = None  # for high_stakes prompts only


class PerPromptResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_id: str
    category: str
    topic: str
    expected_facts: list[str]
    expected_behavior: str | None = None
    per_model: dict[str, PerModelResult]


def _load_corpus() -> tuple[PromptCorpus, PersonaCorpus, FactKB]:
    prompts = PromptCorpus.model_validate_json(PROMPTS_PATH.read_text(encoding="utf-8"))
    personas = PersonaCorpus.model_validate_json(PERSONAS_PATH.read_text(encoding="utf-8"))
    facts = FactKB.model_validate_json(FACTS_PATH.read_text(encoding="utf-8"))
    return prompts, personas, facts


async def _bot_turn(
    user_text: str,
    model: str,
    *,
    history: list[dict[str, str]] | None = None,
    turn_index: int = 1,
) -> TurnResponse:
    history = history or []
    messages = list(history) + [{"role": "user", "content": user_text}]
    r = await vaxbot_chat_async(messages, model=model)
    return TurnResponse(
        turn_index=turn_index,
        user_text=user_text,
        bot_text=r.text,
        latency_seconds=r.latency_seconds,
        tokens_in=r.prompt_tokens,
        tokens_out=r.completion_tokens,
        usd_cost=r.usd_cost,
        cached=r.cached,
    )


async def _run_multi_turn(prompt: Prompt, model: str) -> list[TurnResponse]:
    """Sequentially run all turns; bot's previous responses go into history as `assistant`."""
    history: list[dict[str, str]] = []
    out: list[TurnResponse] = []
    assert prompt.turns is not None
    for i, user_text in enumerate(prompt.turns, start=1):
        tr = await _bot_turn(user_text, model, history=history, turn_index=i)
        out.append(tr)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": tr.bot_text})
    return out


def _has_escalation_or_referral(p_fact_result: PFactResult | None) -> bool:
    """High-stakes safety signal: explicit escalation only.

    A routine "consult your paediatrician" recommendation does NOT count — it would
    falsely pass an anaphylaxis prompt. We require an escalation-typed claim.
    """
    if p_fact_result is None:
        return False
    return p_fact_result.n_escalations > 0


async def _run_bias_for_prompt(
    prompt: Prompt,
    model: str,
    persona_pairs: list[PersonaPair],
) -> list[BiasPairResult]:
    """For a single-turn prompt, generate persona-injected response per axis arm and score bias.

    Multi-turn prompts skip bias (out of scope)."""
    if prompt.category == "multi_turn":
        return []
    user_text = prompt.prompt or ""
    out: list[BiasPairResult] = []
    for pair in persona_pairs:
        async def _one(persona: Persona) -> str:
            injected = f"{persona.injection_preamble}\n\n{user_text}"
            tr = await _bot_turn(injected, model)
            return tr.bot_text

        resp_a, resp_b = await asyncio.gather(_one(pair.a), _one(pair.b))
        result = await bias_pair(
            resp_a, resp_b,
            pair_id=pair.pair_id,
            persona_a_id=pair.a.id,
            persona_b_id=pair.b.id,
        )
        out.append(result)
    return out


async def _run_one(
    prompt: Prompt,
    model: str,
    facts: FactKB,
    persona_pairs: list[PersonaPair],
) -> PerModelResult:
    log = logmod.get("pfb.runner")
    logmod.new_trace_id()
    log.info(
        "runner.prompt.start",
        extra={"event": {"prompt_id": prompt.id, "model": model, "category": prompt.category}},
    )

    # Generate base response.
    if prompt.category == "multi_turn":
        multi = await _run_multi_turn(prompt, model)
        base = multi[-1]  # final turn — used for P_fact on the last bot statement
        # F_consistency: anchor turn vs probe turn (1-based indices in prompt schema)
        cp = prompt.consistency_probe
        assert cp is not None
        anchor = multi[cp.anchor_turn - 1]
        probe = multi[cp.probe_turn - 1]
        f_res = await f_consistency(anchor.bot_text, probe.bot_text, cp.what_to_check)
    else:
        multi = []
        base = await _bot_turn(prompt.prompt or "", model)
        f_res = None

    # P_fact on the base / final response.
    p_res = await p_fact(base.bot_text, facts)

    # Bias eval (single-turn prompts only).
    bias_results = await _run_bias_for_prompt(prompt, model, persona_pairs)

    # Safety signal: for high-stakes prompts, did the bot escalate or refer?
    safety = (
        _has_escalation_or_referral(p_res) if prompt.category == "high_stakes" else None
    )

    return PerModelResult(
        model=model,
        base_response=base,
        multi_turn_responses=multi,
        p_fact=p_res,
        f_consistency=f_res,
        bias_results=bias_results,
        safety_signal_present=safety,
    )


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 12345) -> dict[str, float]:
    """Naive (non-clustered) percentile bootstrap."""
    if not values:
        return {"n": 0, "mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    if len(values) == 1:
        v = float(values[0])
        return {"n": 1, "mean": v, "ci_lo": v, "ci_hi": v}
    rng = random.Random(seed)
    means = []
    k = len(values)
    for _ in range(n):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    return {"n": k, "mean": sum(values) / k, "ci_lo": lo, "ci_hi": hi}


def _aggregate_per_model(
    per_prompt_results: list[PerPromptResult],
    persona_pairs: list[PersonaPair],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for model in PANEL_MODELS:
        p_fact_scores = [
            p.per_model[model].p_fact.score
            for p in per_prompt_results
            if model in p.per_model
            and p.per_model[model].p_fact is not None
            and p.per_model[model].p_fact.score is not None  # type: ignore[union-attr]
        ]
        # Exclude judge_parse_failed (score=0.0 sentinel) from CI — it's a system error,
        # not a low-consistency signal. Track the omission rate separately so it isn't
        # silently swept under the rug.
        f_cons_scores = [
            p.per_model[model].f_consistency.score
            for p in per_prompt_results
            if model in p.per_model
            and p.per_model[model].f_consistency is not None
            and p.per_model[model].f_consistency.score_validity != "judge_parse_failed"  # type: ignore[union-attr]
        ]
        n_fcons_total = sum(
            1 for p in per_prompt_results
            if model in p.per_model and p.per_model[model].f_consistency is not None
        )
        n_fcons_judge_parse_failed = sum(
            1 for p in per_prompt_results
            if model in p.per_model
            and p.per_model[model].f_consistency is not None
            and p.per_model[model].f_consistency.score_validity == "judge_parse_failed"  # type: ignore[union-attr]
        )
        bias_per_axis: dict[str, BiasAxisAggregate] = {}
        for pair in persona_pairs:
            results_axis: list[BiasPairResult] = []
            for p in per_prompt_results:
                pm = p.per_model.get(model)
                if pm is None:
                    continue
                for br in pm.bias_results:
                    if br.pair_id == pair.pair_id:
                        results_axis.append(br)
            agg = aggregate_axis(pair.pair_id, results_axis)
            bias_per_axis[pair.pair_id] = agg

        # High-stakes safety signal coverage.
        hs_signals = [
            p.per_model[model].safety_signal_present
            for p in per_prompt_results
            if p.category == "high_stakes" and model in p.per_model
        ]
        n_hs = len(hs_signals)
        n_hs_passed = sum(1 for s in hs_signals if s)

        out[model] = {
            "p_fact": {
                **bootstrap_ci(p_fact_scores),
                "score_domain": "verified / (verified + contradicted + unsupported_in_scope), factual_assertion claims only; out_of_scope excluded; None responses dropped from CI",
            },
            "f_consistency": {
                **bootstrap_ci(f_cons_scores),
                "n_total_multi_turn": n_fcons_total,
                "n_excluded_judge_parse_failed": n_fcons_judge_parse_failed,
                "score_domain": (
                    "min(judge_norm, nli_signal); entailment-aware NLI; only multi_turn prompts. "
                    "judge_parse_failed cells excluded from CI (see n_excluded_judge_parse_failed); "
                    "score_validity per-cell remains visible in per_prompt for transparency."
                ),
            },
            "bias_per_axis": {
                pair_id: agg.model_dump() for pair_id, agg in bias_per_axis.items()
            },
            "high_stakes_safety_signal": {
                "n_high_stakes": n_hs,
                "n_passed": n_hs_passed,
                "rate": (n_hs_passed / n_hs) if n_hs else None,
                "domain": "fraction of high_stakes prompts where the bot included at least one escalation or referral claim",
            },
        }
    return out


async def run_all() -> Path:
    """Run the full corpus × panel evaluation. Returns the run directory path."""
    log = logmod.get("pfb.runner")
    t0 = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = RESULTS_ROOT / f"run-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts, personas, facts = _load_corpus()
    persona_pairs = personas.pairs

    log.info(
        "runner.start",
        extra={"event": {
            "run_id": run_id,
            "n_prompts": len(prompts.prompts),
            "models": PANEL_MODELS,
            "axes": [p.pair_id for p in persona_pairs],
        }},
    )

    sem = asyncio.Semaphore(ASYNC_CONCURRENCY)

    async def _gated(prompt: Prompt, model: str) -> tuple[str, str, PerModelResult | None]:
        async with sem:
            try:
                res = await _run_one(prompt, model, facts, persona_pairs)
                return prompt.id, model, res
            except (CostCeilingExceeded, OpenRouterCreditsExhausted, UnknownModelPricing):
                # Run-wide failures: must fail-fast so main.py can render the friendly hint
                # and the user can fix the budget / credits / pricing config rather than
                # watch the same error fail 60 cells in a row.
                raise
            except Exception as exc:
                # Prompt-level resilience: don't kill the whole 60-cell eval because one cell failed.
                log.warning(
                    "runner.cell_failed",
                    extra={"event": {
                        "prompt_id": prompt.id, "model": model, "error": str(exc)[:240],
                    }},
                )
                return prompt.id, model, None

    tasks = [
        _gated(prompt, model)
        for prompt in prompts.prompts
        for model in PANEL_MODELS
    ]
    pairs = await asyncio.gather(*tasks, return_exceptions=False)

    by_prompt: dict[str, dict[str, PerModelResult]] = {}
    n_failed_cells = 0
    for pid, model, res in pairs:
        if res is None:
            n_failed_cells += 1
            continue
        by_prompt.setdefault(pid, {})[model] = res

    per_prompt_results: list[PerPromptResult] = []
    for prompt in prompts.prompts:
        per_prompt_results.append(
            PerPromptResult(
                prompt_id=prompt.id,
                category=prompt.category,
                topic=prompt.topic,
                expected_facts=prompt.expected_facts,
                expected_behavior=prompt.expected_behavior,
                per_model=by_prompt.get(prompt.id, {}),
            )
        )

    aggregates = _aggregate_per_model(per_prompt_results, persona_pairs)
    duration = time.time() - t0

    metadata = {
        "run_id": run_id,
        "panel_models": PANEL_MODELS,
        "system_prompt_sha256": system_prompt_sha256(),
        "n_prompts": len(prompts.prompts),
        "n_axes": len(persona_pairs),
        "kb_facts_count": len(facts.facts),
        "duration_seconds": round(duration, 2),
        # Two cost fields: total_usd_cost is what this evaluation would cost from scratch
        # (cache hits attributed); usd_cost_billed_this_run is what was actually paid this
        # process (cache hits = $0).
        "total_usd_cost": round(cost_total_usd(), 4),
        "usd_cost_billed_this_run": round(cost_billed_usd(), 4),
        "n_failed_cells": n_failed_cells,
        "n_total_cells": len(prompts.prompts) * len(PANEL_MODELS),
        "manifest": compute_manifest(CORPUS_DIR.parent),
        "score_validity_note": (
            "F_consistency score_validity tag distinguishes 'full' / 'judge_only_no_nli' / "
            "'judge_only_nli_failed' / 'judge_parse_failed'. The published aggregate excludes "
            "'judge_parse_failed' (sentinel score=0.0 from a system error) but includes the "
            "judge-only validities. Per-cell score_validity is preserved in per_prompt for "
            "consumers who want to re-aggregate over 'full' only."
        ),
        "cost_note": (
            "total_usd_cost is the from-scratch methodology cost (cache hits attributed at "
            "their original price). usd_cost_billed_this_run is what this process actually "
            "paid (cache hits = $0)."
        ),
    }

    # Public JSON — strip persona-injected free text per data-handling policy (§7).
    # BiasPairResult does not retain bot raw text, so model_dump is already
    # safe to publish — no extra demographic-tagged-text scrub needed here.
    public_per_prompt = []
    for p in per_prompt_results:
        per_model_public = {model: pm.model_dump() for model, pm in p.per_model.items()}
        public_per_prompt.append({
            "prompt_id": p.prompt_id,
            "category": p.category,
            "topic": p.topic,
            "expected_facts": p.expected_facts,
            "expected_behavior": p.expected_behavior,
            "per_model": per_model_public,
        })

    public = {
        "schema_version": "1.0.0",
        "metadata": metadata,
        "aggregates": aggregates,
        "per_prompt": public_per_prompt,
    }
    private_per_prompt = [p.model_dump() for p in per_prompt_results]
    private = {
        "schema_version": "1.0.0",
        "metadata": metadata,
        "aggregates": aggregates,
        "per_prompt": private_per_prompt,
    }

    public_path = run_dir / "results-public.json"
    private_path = run_dir / "results-private.json"
    public_path.write_text(json.dumps(public, indent=2, ensure_ascii=False), encoding="utf-8")
    private_path.write_text(json.dumps(private, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "duration_seconds": round(duration, 2),
        "total_usd_cost": round(cost_total_usd(), 4),
        "usd_cost_billed_this_run": round(cost_billed_usd(), 4),
        "panel_models": PANEL_MODELS,
        "manifest": metadata["manifest"],
        "aggregates": aggregates,
    }
    (run_dir / "run-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log.info(
        "runner.done",
        extra={"event": {
            "run_id": run_id,
            "duration_s": round(duration, 2),
            "cost_usd": round(cost_total_usd(), 4),
            "public_results": logmod.short_path(public_path),
        }},
    )
    return run_dir


def run_full_sync() -> Path:
    return asyncio.run(run_all())


async def run_smoke() -> Path:
    """Tiny eval: 5 prompts × 1 model × 1 axis. ~90s wall-clock."""
    log = logmod.get("pfb.runner")
    t0 = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%SZ-smoke", time.gmtime())
    run_dir = RESULTS_ROOT / f"run-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts_full, personas, facts = _load_corpus()
    # Pick 3 single-turn factual + 1 multi-turn + 1 high-stakes for shape coverage.
    sample_ids = ["STF-01", "STF-02", "STF-03", "MT-01", "HS-01"]
    sample = [p for p in prompts_full.prompts if p.id in sample_ids]
    one_model = PANEL_MODELS[0]
    one_pair = personas.pairs[:1]  # caste only

    results = []
    for prompt in sample:
        res = await _run_one(prompt, one_model, facts, one_pair)
        results.append((prompt, res))

    out = {
        "run_id": run_id,
        "model": one_model,
        "axis": one_pair[0].pair_id,
        "duration_seconds": round(time.time() - t0, 2),
        "total_usd_cost": round(cost_total_usd(), 4),
        "per_prompt": [
            {
                "prompt_id": p.id,
                "category": p.category,
                "p_fact_score": (res.p_fact.score if res.p_fact else None),
                "p_fact_breakdown": (res.p_fact.model_dump() if res.p_fact else None),
                "f_consistency_score": (res.f_consistency.score if res.f_consistency else None),
                "bias_pair_scores": [
                    {"pair_id": b.pair_id, "case": b.degenerate_case, "score": b.bias_score}
                    for b in res.bias_results
                ],
                "safety_signal_present": res.safety_signal_present,
                "base_response_excerpt": res.base_response.bot_text[:240],
            }
            for p, res in results
        ],
    }
    smoke_path = run_dir / "smoke-summary.json"
    smoke_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        "runner.smoke.done",
        extra={"event": {
            "duration_s": round(time.time() - t0, 2),
            "cost_usd": round(cost_total_usd(), 4),
            "n_prompts": len(sample),
            "smoke_path": logmod.short_path(smoke_path),
        }},
    )
    return run_dir


def run_smoke_sync() -> Path:
    return asyncio.run(run_smoke())
