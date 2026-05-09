# pfb-eval

A three-axis trustworthiness evaluation harness for **multi-turn** conversational AI in regulated, Indian-context healthcare domains. Demonstrated on a synthetic immunisation chatbot grounded in Government of India and World Health Organization guidance.

## Motivation

Trustworthy conversational AI in regulated domains involves at least three concerns that the evaluation literature usually treats separately:

1. Are the model's claims factually correct against an authoritative knowledge base?
2. Does the model contradict itself across the turns of a conversation?
3. Do recommendations differ across demographic groups when nothing else differs?

Most existing evaluation tools answer one of these on a single response, in a single turn, against a single user persona. The shipped failure modes of conversational AI in healthcare — a chatbot contradicting its own earlier dose-schedule statement, recommending different venues to two equally-situated users — only show up across turns and across paired persona variants. Public issue trackers for the open-source evaluation tools in this space are dominated by single-shot concerns; multi-turn faithfulness and persona-paired fairness are visibly under-addressed. This repository attempts the integration on a deliberately narrow domain (Indian immunisation), with a thirty-prompt corpus and a two-model panel.

## Repository layout

```
.
├── src/
│   ├── eval/                         GENERIC harness (reusable across demos)
│   │   ├── llm_client.py             async OpenRouter client; tenacity retries on
│   │   │                             transient errors; selective fail-fast on 4xx;
│   │   │                             two cost counters (billed vs attributed); typed
│   │   │                             OpenRouterCreditsExhausted / CostCeilingExceeded
│   │   │                             surfaced with friendly hints.
│   │   ├── log.py                    structured logging + JSONL traces + trace IDs.
│   │   ├── cache.py                  on-disk JSON cache; atomic write to avoid corruption.
│   │   ├── manifest.py               SHA-256 of every input artefact; refuses to write
│   │   │                             a degraded manifest (would silently break repro).
│   │   ├── schemas.py                pydantic data validation for corpus + I/O.
│   │   ├── p_fact.py                 Plausibility metric: atomic-claim verification
│   │   │                             against a domain KB; four-way verdict taxonomy.
│   │   ├── f_consistency.py          Faithfulness metric: LLM-judge Likert + DeBERTa
│   │   │                             NLI; conservative min combination; entailment-aware.
│   │   ├── bias.py                   Bias metric: paired persona injection on three
│   │   │                             axes; six-field recommendation comparison.
│   │   ├── runner.py                 orchestrator + naive percentile bootstrap CIs.
│   │   └── judge_prompts/            versioned LLM-judge system prompts.
│   ├── demos/
│   │   └── vaxbot/                   ONE demo: immunisation domain
│   │       ├── system_prompt.md
│   │       ├── client.py
│   │       └── corpus/
│   │           ├── prompts.json      30 hand-curated prompts; 10 are multi-turn.
│   │           ├── personas.json     3 paired persona axes (caste, religion, SES).
│   │           └── facts.json        ~50 atomic UIP/WHO facts as the KB.
│   ├── report/                       Jinja2 template + renderer for docs/index.html.
│   └── main.py                       CLI: smoke / smoke-eval / run / report.
├── tests/                            pytest — schema / cache / cost / aggregation logic.
├── docs/                             GitHub Pages root: live-endpoint report + JSON.
├── models/                           gitignored; IndiCASA encoder weights (optional).
└── results/                          gitignored; cache + traces + per-run JSON.
```

The harness in `src/eval/` makes no domain-specific assumptions. The immunisation chatbot in `src/demos/vaxbot/` is one demonstration of how the harness can be used; swap in any conversational endpoint, knowledge base, and persona library to evaluate a different domain.

## Methodology

For each prompt, the chatbot's response is scored on three independent axes.

- **Plausibility** extracts atomic factual claims from the response, classifies them by claim type, and verifies each `factual_assertion` against an authoritative knowledge base, returning one of four verdicts: `verified`, `contradicted`, `unsupported_in_scope` (a likely hallucination on a covered topic), or `out_of_scope` (a topic the KB does not cover).
- **Faithfulness** runs only on multi-turn conversations: an anchor turn where the chatbot states a fact is paired with a probe turn that tests whether the chatbot's later advice respects that fact, scored by an LLM-judge Likert and an entailment-aware DeBERTa-v3 NLI signal combined with a conservative `min`.
- **Bias** runs paired persona injection on three independent demographic axes (caste, religion, socioeconomic status) where every attribute except the demographic dimension under test is held constant, and scores divergence between the two arms' actionable recommendations on six dimensions.

The Faithfulness axis is what most upstream evaluation tools cannot do: their data models represent a test case as a single `(prompt, response)` tuple with no notion of conversation history, so multi-turn failure modes are invisible to them. Operationalising Faithfulness against a KB-bounded Plausibility check and a persona-paired Bias check on the same conversation is the contribution of this harness.

The full methodology — formulas, edge-case handling, the four-way verdict taxonomy, and per-pair degenerate-case treatment — is documented in the live-endpoint report rendered to `docs/index.html`.

## Live endpoint

The rendered report is published to GitHub Pages and contains the upstream-tool critique, the methodology, results across the panel, and the limitations section. The HTML is fully self-contained — no external scripts, fonts, or styles — so it can be opened directly via `file://` for inspection, served from any static host, or shared as a single file.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv` or see uv docs) and Python 3.11 or newer.

```bash
cp .env.example .env            # then edit .env to set OPENROUTER_API_KEY
uv sync --extra ml --extra dev  # ml extras pull torch + transformers + sentence-transformers
make test                       # ~40 tests covering schema, cache, cost, aggregation logic
```

The harness uses [OpenRouter](https://openrouter.ai/) as a single API gateway — one `OPENROUTER_API_KEY` covers every panel and judge model, both OpenAI-family and Google-family.

For the IndiCASA secondary bias signal (optional, gracefully no-op if absent), download the published encoder weights:

```bash
git lfs install
git lfs clone https://github.com/cerai-iitm/IndiCASA /tmp/IndiCASA
cp /tmp/IndiCASA/BiasMetric/best_contrastive_model/best_model.pth ./models/
```

If the encoder is missing, the bias metric falls back to recommendation-divergence only.

## Run

```bash
make smoke         # one factual + one high-stakes prompt against both panel models
make smoke-eval    # 5 prompts × 1 model × 1 axis through the full pipeline (~2 min)
make run           # 30 prompts × 2 models × 3 axes — full evaluation (~10-20 min)
make report        # render docs/index.html + docs/results.json from the latest full run
```

The cost ceiling defaults to USD 25 (override with `PFB_MAX_USD`). Two cost counters are surfaced in the run summary: a *billed* cost (what this process actually paid the provider, with cache hits at zero — used for ceiling enforcement) and an *attributed* cost (the full from-scratch cost of the methodology, with cache hits priced at their original cost — used for accurate reporting so a cached re-run does not appear free). Cache writes to `results/cache/` keyed on a SHA-256 of (model, canonical messages, parameters); set `PFB_NO_CACHE=1` to bypass.

## Engineering

- 40+ pytest covering schema validation, cache key determinism, cost calculation, cost-ceiling enforcement (with cached calls correctly NOT counted), and the pure-function aggregation logic of every metric.
- `tenacity`-driven retries on transient API failures (network, timeout, rate-limit, 5xx) with exponential backoff and jitter; client errors (auth, bad request, 402 / out of credits) fail fast with a typed exception and a one-line user hint surfaced via the CLI.
- `ruff` lint and format configured in `pyproject.toml`; pre-commit hooks at `.pre-commit-config.yaml` enforce ruff, trailing-whitespace, end-of-file fixer, large-file blocking, and private-key detection.
- Reproducibility: every run embeds in its summary a SHA-256 manifest of the chatbot system prompt, every judge prompt, and every corpus file. The manifest writer refuses to write a degraded manifest (missing system prompt or judge prompts) so an unreproducible run cannot silently appear reproducible.
- Cache writes are atomic (temp file plus `os.replace`) so an interrupted process cannot leave a half-written JSON file that the next read silently swallows.
- Path hygiene: structured log events use project-relative paths via a `short_path` helper, so absolute filesystem paths do not leak into traces, exception messages, or shared output.

## Citations

- **LExT** — Shailya, Rajpal, Krishnan, Ravindran. *Towards Evaluating Trustworthiness of Natural Language Explanations.* FAccT 2025. [arXiv:2504.06227](https://arxiv.org/abs/2504.06227). The two-axis Plausibility plus Faithfulness decomposition principle, applied to single-shot QA in the original work; this harness extends it to multi-turn conversational evaluation.
- **IndiCASA** — Santhosh et al. *IndiBias-based Contextually Aligned Stereotypes and Anti-Stereotypes Dataset.* 2025. [arXiv:2510.02742](https://arxiv.org/abs/2510.02742). The Indian-context bias methodology; the encoder weights are reused as a secondary similarity signal, and the contrastive-pair design is extended from masked-template completion to free-text recommendation comparison.
- **DeepEval** — [confident-ai/deepeval](https://github.com/confident-ai/deepeval). Methodological precedent for `TurnFaithfulnessMetric` and `ConversationalGEval`. The within-conversation faithfulness idea is re-implemented here so that the exact judge prompt, NLI head, and combination rule are documented in this repository.

## Limitations

The harness is a methodology demonstration, not a production-grade benchmark.

- Thirty hand-curated prompts is illustrative for cross-model trends; specific cross-model rank-orderings should not be over-interpreted, and confidence intervals are naive percentile bootstrap rather than cluster-bootstrap.
- All three metrics rely on LLM-judge components. Judge calibration against expert-graded ground truth is a research programme of weeks-to-months and is not undertaken here. Same-family contamination is acknowledged: the assistant building the harness is an Anthropic model, while the primary verification judge is Google-family.
- Three of the five IndiCASA bias axes are evaluated. Gender and disability axes are named as future work.
- Personas hold their stated healthcare routine constant across both arms of every pair to isolate demographic-marker effects from access-aware tailoring effects; this introduces a deliberate realism cost named in the rendered report's Limitations section.
- The chatbot system prompt and the corpus have not been reviewed by clinical or immunological domain experts. The chatbot is a research artefact and must not be deployed without independent clinical review, regulatory sign-off, content-safety guardrails, an escalation flow, and IRB or equivalent oversight. The first line of every chatbot system prompt and the first line of the live-endpoint report state this.

## Built with AI assistance

Code scaffolding, dataset adapters, judge prompts, and the report renderer were drafted with Claude (Anthropic) under direction by the project author, who retained methodology, scope, and load-bearing design decisions.

## License

MIT. See `LICENSE`.
