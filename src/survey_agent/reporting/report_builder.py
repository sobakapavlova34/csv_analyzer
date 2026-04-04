"""
Сборка читаемого HTML-отчёта по каталогу прогона (без знания Python).

Открыть ``report/index.html`` в браузере; для PDF: Печать → «Сохранить как PDF».
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from survey_agent.pipeline.hypothesis_tests import link_hypothesis_to_battery_ids
from survey_agent.reporting.figures import build_all_figures


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _reliability_note(p: float | None) -> str:
    if p is None or p != p:
        return "—"
    if p < 0.001:
        return "очень сильная статистическая опора (p < 0.001)"
    if p < 0.01:
        return "сильная опора (p < 0.01)"
    if p < 0.05:
        return "умеренная опора (p < 0.05); эффект может быть случайным при множестве тестов"
    return "слабая опора (p ≥ 0.05); возможен шум или малая выборка в подгруппах"


def build_html_report(run_dir: Path | str) -> Path:
    run_dir = Path(run_dir).resolve()
    pre = run_dir / "preprocess"
    eda = run_dir / "eda"
    hyp = run_dir / "hypotheses"
    tst = run_dir / "tests"

    prep = _load_json(pre / "report.json") or {}
    cluster_story = _load_json(eda / "cluster_story.json") or {}
    hyp_data = _load_json(hyp / "hypotheses.json") or {}
    stat = _load_json(tst / "statistical_report.json") or {}
    manifest = _load_json(run_dir / "run_manifest.json") or {}

    figures = build_all_figures(run_dir, cluster_story if cluster_story else None)

    battery = stat.get("battery") or []
    battery_by_id: dict[str, dict[str, Any]] = {}
    for b in battery:
        bid = b.get("id")
        if bid:
            battery_by_id[str(bid)] = b
    available_ids = set(battery_by_id.keys())

    def _row_for_battery_item(b: dict[str, Any]) -> dict[str, Any]:
        p = b.get("pvalue")
        return {
            "id": b.get("id", ""),
            "question": b.get("question", ""),
            "test": b.get("test", ""),
            "pvalue": None if p is None or p != p else round(float(p), 6),
            "effect": _effect_summary(b),
            "reliability": _reliability_note(float(p) if p is not None and p == p else None),
        }

    hyp_list = hyp_data.get("hypotheses") or []
    if isinstance(hyp_list, dict):
        hyp_list = list(hyp_list.values())

    # Не используем hypothesis_to_test_map.json: файл мог устареть после правки hypotheses.json
    # без повторного run_statistical_tests — связи всегда из текущих гипотез + актуальной батареи.
    hypotheses_with_tests: list[dict[str, Any]] = []
    for i, h in enumerate(hyp_list):
        if not isinstance(h, dict):
            hypotheses_with_tests.append({"raw": h, "tests": []})
            continue
        tids = link_hypothesis_to_battery_ids(h, available_ids)
        test_rows = [_row_for_battery_item(battery_by_id[tid]) for tid in tids if tid in battery_by_id]
        hypotheses_with_tests.append({**h, "_index": i, "tests": test_rows})

    # При сбое LLM в этом прогоне hypotheses.json всё равно хранит source=error — отчёт только читает файл.
    # Показываем ориентиры из cluster_story, чтобы блок не был пустым.
    fallback_hints: list[str] = list(cluster_story.get("hints_for_hypotheses") or [])
    fallback_from_tests: list[str] = []
    for b in battery[:8]:
        q = b.get("question")
        tid = b.get("id")
        if q and tid:
            fallback_from_tests.append(f"[{tid}] {q}")

    env = Environment(
        loader=PackageLoader("survey_agent.reporting", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template("report.html.j2")

    hyp_path = hyp / "hypotheses.json"
    hyp_count = len(hyp_list) if isinstance(hyp_list, list) else 0

    ctx = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_slug": run_dir.name,
        "hypothesis_json_path": str(hyp_path),
        "hypothesis_count": hyp_count,
        "csv_source": manifest.get("csv_source", "—"),
        "n_rows": prep.get("n_rows"),
        "n_columns": prep.get("n_columns"),
        "cluster_story": cluster_story,
        "hypotheses": hyp_list,
        "hypotheses_with_tests": hypotheses_with_tests,
        "hyp_source": hyp_data.get("source", ""),
        "hyp_error": hyp_data.get("error"),
        "hypothesis_fallback_hints": fallback_hints,
        "hypothesis_fallback_tests": fallback_from_tests,
        "has_hypothesis_text": bool(hyp_list),
        "hypotheses_json_found": hyp_path.is_file(),
        "figures": figures,
        "has_figures": bool(figures),
    }

    html = tpl.render(**ctx)
    out_dir = run_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _effect_summary(b: dict[str, Any]) -> str:
    if "cohens_d" in b:
        d = b["cohens_d"]
        if d == d:
            return f"Cohen's d ≈ {float(d):.3f}"
    if "cramers_v" in b:
        return f"Cramér's V ≈ {float(b['cramers_v']):.3f}"
    if "r" in b:
        lo, hi = b.get("ci_r_95") or (None, None)
        if lo is not None and hi is not None:
            return f"r ≈ {b['r']}, 95% ДИ для r: [{lo}; {hi}]"
        return f"r ≈ {b.get('r')}"
    if "mean_difference" in b:
        ci = b.get("ci_mean_diff_95")
        if ci and len(ci) == 2:
            return f"разница средних {float(b['mean_difference']):.3f}, 95% ДИ: [{float(ci[0]):.3f}; {float(ci[1]):.3f}]"
    return "—"

