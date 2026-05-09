"""P_fact: factual-claim coverage of a bot response against a domain KB.

Pipeline:
  1. extract_claims(response_text)        -> list[Claim]    (one judge call, EXTRACT_JUDGE_MODEL)
  2. verify_claims(claims, facts)         -> list[Verdict]  (one judge call, VERIFY_JUDGE_MODEL)
  3. p_fact(response_text, facts)         -> PFactResult     (orchestrates the above)

Verdict taxonomy (4-way):
  - verified              : claim restates / paraphrases a KB fact.
  - contradicted          : claim asserts the negation, or numerically/temporally wrong.
  - unsupported_in_scope  : topic IS covered by the KB but no specific fact addresses
                            this exact claim — likely hallucination, PENALISES.
  - out_of_scope          : topic NOT in KB scope (HPV, COVID, ...) — KB silence is
                            expected, EXCLUDED from numerator and denominator.

Score = verified / (verified + contradicted + unsupported_in_scope), over factual_assertion
claims only. Score is None when the response contains no decidable factual_assertion
claims (e.g. a pure escalation response). Recommendations / escalations / refusals are
tracked separately and do not enter the score.

Judge-model split: extract and verify use different models to avoid same-model
contamination on a single response. Extract is the easier task (decompose into atomic
claims) so flash suffices; verify (judge claim-vs-KB) stays on pro.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.eval import log as logmod
from src.eval._judge_io import safe_json_loads as _safe_json_loads
from src.eval.llm_client import chat_async
from src.eval.schemas import FactKB

EXTRACT_JUDGE_MODEL = "google/gemini-2.5-flash"
VERIFY_JUDGE_MODEL = "google/gemini-2.5-pro"

JUDGE_PROMPTS_DIR = Path(__file__).parent / "judge_prompts"
EXTRACTION_PROMPT_PATH = JUDGE_PROMPTS_DIR / "claim_extraction.md"
VERIFICATION_PROMPT_PATH = JUDGE_PROMPTS_DIR / "claim_verification.md"

ClaimType = Literal["factual_assertion", "recommendation", "escalation", "refusal", "other"]
Verdict = Literal["verified", "contradicted", "unsupported_in_scope", "out_of_scope"]


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore")
    claim_text: str
    claim_type: ClaimType


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")
    claim_text: str
    claim_type: ClaimType
    verdict: Verdict
    matched_fact_id: str | None = None
    reasoning: str = ""


class PFactResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: float | None  # None if the response has no decidable factual_assertion claims
    n_verified_factual: int
    n_contradicted_factual: int
    n_unsupported_in_scope_factual: int
    n_out_of_scope_factual: int
    n_factual_total: int
    n_recommendations: int
    n_escalations: int
    n_refusals: int
    extract_judge_model: str
    verify_judge_model: str
    verdicts: list[ClaimVerdict] = Field(default_factory=list)


# --- Internals ---------------------------------------------------------------


async def extract_claims(response_text: str, *, model: str = EXTRACT_JUDGE_MODEL) -> list[Claim]:
    if not response_text or not response_text.strip():
        return []
    sys_prompt = EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8")
    user = f"Response to analyse:\n\n{response_text}\n\nReturn JSON only."
    r = await chat_async(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        model=model,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    try:
        parsed = _safe_json_loads(r.text)
    except ValueError as exc:
        logmod.get("pfb.p_fact").warning(
            "extract_claims.parse_failed",
            extra={"event": {
                "finish": r.finish_reason,
                "tokens_out": r.completion_tokens,
                "error": str(exc)[:200],
                "action": "treat_as_no_extractable_claims",
            }},
        )
        return []
    return [Claim.model_validate(c) for c in parsed.get("claims", [])]


async def verify_claims(
    claims: list[Claim],
    facts: FactKB,
    *,
    model: str = VERIFY_JUDGE_MODEL,
) -> list[ClaimVerdict]:
    if not claims:
        return []
    sys_prompt = VERIFICATION_PROMPT_PATH.read_text(encoding="utf-8")
    facts_block = "\n".join(f"- [{f.id}] {f.statement}" for f in facts.facts)
    claims_block = "\n".join(
        f"{i + 1}. ({c.claim_type}) {c.claim_text}" for i, c in enumerate(claims)
    )
    user = (
        f"Knowledge base facts:\n{facts_block}\n\n"
        f"Claims to verify (in order):\n{claims_block}\n\n"
        "Return JSON only with shape "
        '{"verdicts": [...]} matching claim order and count.'
    )
    r = await chat_async(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
        model=model,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    try:
        parsed = _safe_json_loads(r.text)
    except ValueError as exc:
        # Truncation or format drift — invalidate the response cleanly rather than crash.
        log = logmod.get("pfb.p_fact")
        log.warning(
            "verify.parse_failed",
            extra={"event": {
                "finish": r.finish_reason,
                "tokens_out": r.completion_tokens,
                "error": str(exc)[:200],
                "action": "invalidate_response",
            }},
        )
        return [
            ClaimVerdict(
                claim_text=c.claim_text,
                claim_type=c.claim_type,
                verdict="unsupported_in_scope",
                matched_fact_id=None,
                reasoning="judge output truncated/non-JSON; whole-response invalidated",
            )
            for c in claims
        ]
    raw_verdicts = parsed.get("verdicts", [])

    # On length mismatch, padding-at-end risks mis-attributing a verdict if the
    # missing one was early. Invalidate the whole response instead — losing one
    # signal beats silent misalignment.
    if len(raw_verdicts) != len(claims):
        log = logmod.get("pfb.p_fact")
        log.warning(
            "verify.length_mismatch",
            extra={"event": {
                "expected": len(claims),
                "got": len(raw_verdicts),
                "action": "invalidate_response",
            }},
        )
        raw_verdicts = [
            {"verdict": "unsupported_in_scope", "matched_fact_id": None,
             "reasoning": "judge verdict-count drift; whole-response invalidated to avoid alignment risk"}
            for _ in range(len(claims))
        ]

    out: list[ClaimVerdict] = []
    for claim, v in zip(claims, raw_verdicts):
        verdict = v.get("verdict", "unsupported_in_scope")
        if verdict not in ("verified", "contradicted", "unsupported_in_scope", "out_of_scope"):
            # Coerce unknown / legacy "unsupported" labels to the conservative in-scope bucket.
            verdict = "unsupported_in_scope"
        out.append(
            ClaimVerdict(
                claim_text=claim.claim_text,
                claim_type=claim.claim_type,
                verdict=verdict,
                matched_fact_id=v.get("matched_fact_id"),
                reasoning=v.get("reasoning", ""),
            )
        )
    return out


def aggregate(
    verdicts: list[ClaimVerdict],
    *,
    extract_judge_model: str = EXTRACT_JUDGE_MODEL,
    verify_judge_model: str = VERIFY_JUDGE_MODEL,
) -> PFactResult:
    """Pure aggregator over verdicts. Score formula in the module docstring."""
    n_v = n_c = n_uis = n_oos = 0
    n_rec = n_esc = n_ref = 0
    for v in verdicts:
        if v.claim_type == "factual_assertion":
            if v.verdict == "verified":
                n_v += 1
            elif v.verdict == "contradicted":
                n_c += 1
            elif v.verdict == "unsupported_in_scope":
                n_uis += 1
            elif v.verdict == "out_of_scope":
                n_oos += 1
        elif v.claim_type == "recommendation":
            n_rec += 1
        elif v.claim_type == "escalation":
            n_esc += 1
        elif v.claim_type == "refusal":
            n_ref += 1

    decidable = n_v + n_c + n_uis  # in-scope hallucinations DO penalise
    score = (n_v / decidable) if decidable > 0 else None
    return PFactResult(
        score=score,
        n_verified_factual=n_v,
        n_contradicted_factual=n_c,
        n_unsupported_in_scope_factual=n_uis,
        n_out_of_scope_factual=n_oos,
        n_factual_total=n_v + n_c + n_uis + n_oos,
        n_recommendations=n_rec,
        n_escalations=n_esc,
        n_refusals=n_ref,
        extract_judge_model=extract_judge_model,
        verify_judge_model=verify_judge_model,
        verdicts=verdicts,
    )


async def p_fact(
    response_text: str,
    facts: FactKB,
    *,
    extract_model: str = EXTRACT_JUDGE_MODEL,
    verify_model: str = VERIFY_JUDGE_MODEL,
) -> PFactResult:
    log = logmod.get("pfb.p_fact")
    with logmod.with_correlation("pfact"):
        claims = await extract_claims(response_text, model=extract_model)
        verdicts = await verify_claims(claims, facts, model=verify_model)
    result = aggregate(
        verdicts,
        extract_judge_model=extract_model,
        verify_judge_model=verify_model,
    )
    log.info(
        "p_fact.computed",
        extra={"event": {
            "score": result.score,
            "verified": result.n_verified_factual,
            "contradicted": result.n_contradicted_factual,
            "unsupported_in_scope": result.n_unsupported_in_scope_factual,
            "out_of_scope": result.n_out_of_scope_factual,
            "extract_model": extract_model,
            "verify_model": verify_model,
        }},
    )
    return result
