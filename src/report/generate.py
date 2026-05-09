"""Render docs/index.html (the live endpoint) and docs/results.json from the latest full run."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.eval import log as logmod


THIS_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = THIS_DIR / "templates"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"


def find_latest_run() -> Path:
    candidates = [
        p for p in RESULTS_ROOT.glob("run-*")
        if p.is_dir() and not p.name.endswith("-smoke")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no full run found in {logmod.short_path(RESULTS_ROOT)}; run `make run` first"
        )
    return sorted(candidates, key=lambda p: p.name, reverse=True)[0]


def _select_cases(per_prompt: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Pick a handful of cases worth highlighting in the report."""
    cross_model_disagreements = []
    high_bias = []
    failed_safety = []
    low_consistency = []

    for p in per_prompt:
        models = list(p.get("per_model", {}).keys())
        if len(models) >= 2:
            scores = []
            for m in models:
                pf = p["per_model"][m].get("p_fact") or {}
                s = pf.get("score")
                if s is not None:
                    scores.append((m, s))
            if len(scores) == 2 and abs(scores[0][1] - scores[1][1]) >= 0.5:
                cross_model_disagreements.append({
                    "prompt_id": p["prompt_id"],
                    "topic": p["topic"],
                    "scores": scores,
                })

        for m, pm in p.get("per_model", {}).items():
            for br in pm.get("bias_results", []):
                if br.get("bias_score") is not None and br["bias_score"] >= 0.5:
                    high_bias.append({
                        "prompt_id": p["prompt_id"],
                        "topic": p["topic"],
                        "model": m,
                        "axis": br["pair_id"],
                        "score": br["bias_score"],
                        "case": br["degenerate_case"],
                    })

        if p["category"] == "high_stakes":
            for m, pm in p.get("per_model", {}).items():
                if pm.get("safety_signal_present") is False:
                    failed_safety.append({
                        "prompt_id": p["prompt_id"],
                        "topic": p["topic"],
                        "model": m,
                        "expected_behavior": p.get("expected_behavior"),
                    })

        if p["category"] == "multi_turn":
            for m, pm in p.get("per_model", {}).items():
                fc = pm.get("f_consistency") or {}
                s = fc.get("score")
                # Skip judge_parse_failed: score=0.0 is a sentinel for system error,
                # not a low-consistency signal — surfacing it would misrepresent the bot.
                if (
                    s is not None
                    and s < 0.5
                    and fc.get("score_validity") != "judge_parse_failed"
                ):
                    low_consistency.append({
                        "prompt_id": p["prompt_id"],
                        "topic": p["topic"],
                        "model": m,
                        "score": s,
                        "validity": fc.get("score_validity"),
                        "judge_reasoning": fc.get("judge_reasoning", "")[:240],
                    })

    return {
        "cross_model_disagreements": cross_model_disagreements[:5],
        "high_bias": sorted(high_bias, key=lambda x: -x["score"])[:5],
        "failed_safety": failed_safety,
        "low_consistency": sorted(low_consistency, key=lambda x: x["score"])[:5],
    }


def render(run_dir: Path | None = None) -> dict[str, Path]:
    log = logmod.get("pfb.report")
    run_dir = run_dir or find_latest_run()
    public_json = run_dir / "results-public.json"
    if not public_json.exists():
        raise FileNotFoundError(f"missing {logmod.short_path(public_json)}")

    public = json.loads(public_json.read_text(encoding="utf-8"))
    cases = _select_cases(public.get("per_prompt", []))

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    ctx = {
        "metadata": public["metadata"],
        "aggregates": public["aggregates"],
        "per_prompt": public["per_prompt"],
        "cases": cases,
        "render_iso": time.strftime("%Y-%m-%d", time.gmtime()),
        "n_prompts_total": public["metadata"]["n_prompts"],
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    template = env.get_template("index.html.j2")
    html = template.render(**ctx)
    index_path = DOCS_DIR / "index.html"
    index_path.write_text(html, encoding="utf-8")

    # Public-safe results.json — same shape as the run's results-public.json (already redacted).
    results_path = DOCS_DIR / "results.json"
    results_path.write_text(json.dumps(public, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(
        "report.rendered",
        extra={"event": {
            "run_dir": logmod.short_path(run_dir),
            "index": logmod.short_path(index_path),
            "results": logmod.short_path(results_path),
        }},
    )

    return {"index": index_path, "results": results_path}


def render_sync() -> dict[str, Path]:
    return render()
