"""F_consistency: within-conversation self-consistency.

Two-signal aggregate:
  1. LLM-judge Likert 1-5 (via OpenRouter).
  2. DeBERTa-v3 NLI entailment (off-the-shelf HuggingFace), if the ml extras are installed.

Combination:
  - judge_norm = (likert - 1) / 4                                              in [0, 1]
  - nli_signal = clip(P(entailment) + 0.5 * P(neutral) - P(contradiction), 0, 1)
        Entailment-aware: pure-neutral evasive responses score ~0.5, not ~0.95.
  - score      = min(judge_norm, nli_signal)
        Conservative and symmetric. Both signals are returned separately so the
        disagreement direction is visible.

score_validity tags how comparable a given score is across the corpus:
  - "full"                  — both judge and NLI signals available.
  - "judge_only_no_nli"     — `transformers` not installed; NLI dropped gracefully.
  - "judge_only_nli_failed" — NLI installed but the prediction call failed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.eval import log as logmod
from src.eval._judge_io import safe_json_loads as _safe_json_loads
from src.eval.llm_client import chat_async


JUDGE_MODEL = "google/gemini-2.5-pro"
JUDGE_PROMPT_PATH = Path(__file__).parent / "judge_prompts" / "consistency_likert.md"
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


ScoreValidity = Literal[
    "full",
    "judge_only_no_nli",
    "judge_only_nli_failed",
    "judge_parse_failed",
]


class FConsistencyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float
    judge_likert: int
    judge_norm: float
    judge_reasoning: str
    nli_p_contradiction: float | None
    nli_p_entailment: float | None
    nli_p_neutral: float | None
    nli_signal: float | None
    score_validity: ScoreValidity
    judge_model: str


# --- NLI (lazy-loaded; gracefully missing if ml extras not installed) --------

_nli_pipeline = None
_nli_unavailable_reason: str | None = None


def _get_nli():
    """Return a (premise, hypothesis) -> {label: prob} callable, or None if unavailable."""
    global _nli_pipeline, _nli_unavailable_reason
    if _nli_pipeline is not None:
        return _nli_pipeline
    if _nli_unavailable_reason is not None:
        return None
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:
        _nli_unavailable_reason = f"transformers not installed: {exc}"
        logmod.get("pfb.f_cons").warning(
            "nli.unavailable",
            extra={"event": {"reason": _nli_unavailable_reason}},
        )
        return None

    try:
        # text-classification with text+text_pair is more stable across MNLI heads
        # than the dedicated 'zero-shot' pipeline.
        clf = pipeline("text-classification", model=NLI_MODEL_NAME, top_k=None)
    except Exception as exc:
        _nli_unavailable_reason = f"nli model load failed: {exc}"
        logmod.get("pfb.f_cons").warning(
            "nli.unavailable",
            extra={"event": {"reason": _nli_unavailable_reason}},
        )
        return None

    # Some MNLI heads return canonical labels, others return LABEL_0/1/2.
    LABEL_INDEX = {
        "label_0": "contradiction",
        "label_1": "neutral",
        "label_2": "entailment",
    }

    def _normalise_label(raw: str) -> str:
        s = raw.strip().lower()
        return LABEL_INDEX.get(s, s)

    def _predict(premise: str, hypothesis: str) -> dict[str, float]:
        scores = clf({"text": premise, "text_pair": hypothesis})
        # Pipeline returns either [[{label, score}, ...]] (top_k=None) or [{label, score}, ...].
        rows = scores[0] if isinstance(scores, list) and scores and isinstance(scores[0], list) else scores
        return {_normalise_label(row["label"]): float(row["score"]) for row in rows if "label" in row}

    _nli_pipeline = _predict
    return _predict


async def judge_likert(
    anchor_text: str,
    probe_text: str,
    what_to_check: str,
    *,
    model: str = JUDGE_MODEL,
) -> tuple[int, str]:
    sys_prompt = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    user = (
        f"Anchor response (earlier turn):\n---\n{anchor_text}\n---\n\n"
        f"Probe response (later turn):\n---\n{probe_text}\n---\n\n"
        f"What to check: {what_to_check}\n\n"
        "Return JSON only."
    )
    r = await chat_async(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        model=model,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )
    try:
        parsed = _safe_json_loads(r.text)
    except ValueError as exc:
        logmod.get("pfb.f_cons").warning(
            "judge_likert.parse_failed",
            extra={"event": {"finish": r.finish_reason, "tokens_out": r.completion_tokens, "error": str(exc)[:200]}},
        )
        # Sentinel 0 (out-of-band; valid likert is 1-5). f_consistency() maps this
        # to score_validity="judge_parse_failed" and excludes from aggregates.
        return 0, "judge JSON parse failed; result will be tagged judge_parse_failed"
    raw_score = parsed.get("score")
    # Missing or non-coercible score: tag as parse_failed rather than silently
    # defaulting to 3 (which maps to a misleading judge_norm=0.5).
    if raw_score is None:
        logmod.get("pfb.f_cons").warning(
            "judge_likert.missing_score",
            extra={"event": {
                "finish": r.finish_reason,
                "tokens_out": r.completion_tokens,
                "parsed_keys": list(parsed.keys())[:10],
            }},
        )
        return 0, "judge JSON parsed but missing 'score' field; tagged judge_parse_failed"
    try:
        score = int(raw_score)
    except (TypeError, ValueError) as exc:
        logmod.get("pfb.f_cons").warning(
            "judge_likert.score_not_int",
            extra={"event": {
                "finish": r.finish_reason,
                "raw_score": str(raw_score)[:60],
                "error": str(exc)[:200],
            }},
        )
        return 0, f"judge returned non-int score {raw_score!r}; tagged judge_parse_failed"
    score = max(1, min(5, score))
    return score, parsed.get("reasoning", "")


def combine(
    judge_likert_score: int,
    nli_p_contradiction: float | None,
    nli_p_entailment: float | None,
    nli_p_neutral: float | None = None,
) -> tuple[float, ScoreValidity, float, float | None]:
    """Pure aggregator: returns (score, score_validity, judge_norm, nli_signal).

    The "judge_only_*" availability split (no_nli vs nli_failed) is decided by the
    caller — this function only reports whether NLI signals were provided.
    """
    judge_norm = (judge_likert_score - 1) / 4.0
    if nli_p_contradiction is None:
        return judge_norm, "judge_only_no_nli", judge_norm, None
    p_e = nli_p_entailment or 0.0
    p_n = nli_p_neutral or 0.0
    p_c = nli_p_contradiction
    nli_signal = max(0.0, min(1.0, p_e + 0.5 * p_n - p_c))
    score = min(judge_norm, nli_signal)
    return score, "full", judge_norm, nli_signal


async def f_consistency(
    anchor_text: str,
    probe_text: str,
    what_to_check: str,
    *,
    judge_model: str = JUDGE_MODEL,
) -> FConsistencyResult:
    log = logmod.get("pfb.f_cons")
    nli_predict_failed = False
    with logmod.with_correlation("fcons"):
        likert, reasoning = await judge_likert(
            anchor_text, probe_text, what_to_check, model=judge_model
        )

        nli = _get_nli()
        p_c = p_e = p_n = None
        if nli is not None:
            try:
                probs = nli(anchor_text, probe_text)
                p_c = probs.get("contradiction")
                p_e = probs.get("entailment")
                p_n = probs.get("neutral")
            except Exception as exc:
                nli_predict_failed = True
                log.warning(
                    "nli.predict_failed",
                    extra={"event": {"error": str(exc)[:200]}},
                )

    # likert == 0 is the sentinel from judge_likert() for a JSON parse failure.
    if likert == 0:
        score = 0.0
        score_validity: ScoreValidity = "judge_parse_failed"
        judge_norm = 0.0
        nli_signal = None if p_c is None else max(0.0, min(1.0, (p_e or 0.0) + 0.5 * (p_n or 0.0) - p_c))
    else:
        score, base_validity, judge_norm, nli_signal = combine(likert, p_c, p_e, p_n)

        if base_validity == "full":
            score_validity = "full"
        elif nli_predict_failed:
            score_validity = "judge_only_nli_failed"
        else:
            score_validity = "judge_only_no_nli"

    result = FConsistencyResult(
        score=score,
        judge_likert=likert,
        judge_norm=judge_norm,
        judge_reasoning=reasoning,
        nli_p_contradiction=p_c,
        nli_p_entailment=p_e,
        nli_p_neutral=p_n,
        nli_signal=nli_signal,
        score_validity=score_validity,
        judge_model=judge_model,
    )
    log.info(
        "f_consistency.computed",
        extra={"event": {
            "score": result.score,
            "likert": result.judge_likert,
            "score_validity": result.score_validity,
            "judge_norm": round(result.judge_norm, 3),
            "nli_signal": round(result.nli_signal, 3) if result.nli_signal is not None else None,
        }},
    )
    return result
