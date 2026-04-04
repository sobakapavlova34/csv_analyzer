"""
Проверка содержательных «сюжетов» данных: тесты, p-value, эффекты, ДИ.

Пишет ``tests/statistical_report.json`` и ``tests/statistical_report.md`` — пригодно для отчёта по ТЗ.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _pearson_r_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n < 4 or r != r:
        return float("nan"), float("nan")
    z = float(np.arctanh(r))
    se = 1.0 / np.sqrt(n - 3)
    zc = float(stats.norm.ppf(1 - alpha / 2))
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    n1, n2 = len(x), len(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def _welch_ttest_full(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 3 or len(y) < 3:
        return {"error": "мало наблюдений"}
    t_stat, p = stats.ttest_ind(x, y, equal_var=False)
    m1, m2 = float(np.mean(x)), float(np.mean(y))
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    n1, n2 = len(x), len(y)
    se = float(np.sqrt(v1 / n1 + v2 / n2))
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tc = float(stats.t.ppf(1 - alpha / 2, df))
    diff = m1 - m2
    return {
        "test": "Welch t-test (независимые выборки)",
        "mean_x": m1,
        "mean_y": m2,
        "mean_difference": diff,
        "statistic": float(t_stat),
        "pvalue": float(p),
        "df_welch": float(df),
        "ci_mean_diff_95": [float(diff - tc * se), float(diff + tc * se)],
        "cohens_d": _cohens_d(x, y),
        "n_x": n1,
        "n_y": n2,
    }


def _mannwhitney_full(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 3 or len(y) < 3:
        return {"error": "мало наблюдений"}
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    # rank-biserial correlation as effect size (optional)
    rbc = 1 - 2 * u / (len(x) * len(y))
    return {
        "test": "Mann–Whitney U",
        "statistic": float(u),
        "pvalue": float(p),
        "median_x": float(np.median(x)),
        "median_y": float(np.median(y)),
        "rank_biserial_approx": float(rbc),
        "n_x": len(x),
        "n_y": len(y),
    }


def _chi2_cramers_v(table: pd.DataFrame) -> dict[str, Any]:
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    v = float(np.sqrt(chi2 / (n * (min(k - 1, r - 1)))))
    return {
        "test": "Chi-square независимости",
        "chi2": float(chi2),
        "pvalue": float(p),
        "dof": int(dof),
        "cramers_v": v,
        "n": int(n),
        "table_shape": list(table.shape),
    }


def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pearson_pair_row(
    df_enc: pd.DataFrame, col_a: str, col_b: str, test_id: str, question: str
) -> dict[str, Any] | None:
    """Корреляция Пирсона между двумя столбцами trust_* / encoded."""
    if col_a not in df_enc.columns or col_b not in df_enc.columns:
        return None
    pair = df_enc[[col_a, col_b]].dropna()
    if len(pair) < 30:
        return None
    r = float(pair[col_a].corr(pair[col_b]))
    lo, hi = _pearson_r_ci(r, len(pair))
    t_r = r * np.sqrt((len(pair) - 2) / max(1e-9, 1 - r**2))
    p_r = 2 * (1 - stats.t.cdf(abs(t_r), len(pair) - 2))
    return {
        "id": test_id,
        "question": question,
        "test": "Pearson correlation",
        "r": round(r, 4),
        "pvalue": float(p_r),
        "n": len(pair),
        "ci_r_95": [round(lo, 4), round(hi, 4)],
        "interpretation": "Связь двух шкал доверия; не путать с причинностью.",
    }


def link_hypothesis_to_battery_ids(h: dict[str, Any], available_ids: set[str]) -> list[str]:
    """
    Сопоставляет одну гипотезу с id тестов из батареи по variables_involved и формулировке.
    """
    v = set(h.get("variables_involved") or [])
    title = (h.get("title") or "").lower()
    stmt = (h.get("statement") or "").lower()
    text = f"{title} {stmt}"

    church_focus = "trust_church" in v or "церкв" in text
    cluster_focus = "кластер" in text or "cluster" in text

    out: list[str] = []

    def add(tid: str) -> None:
        if tid in available_ids and tid not in out:
            out.append(tid)

    # Гипотеза про кластеры: LLM часто перечисляет все trust_* — не цеплять T2/T6 и т.п.
    if cluster_focus:
        add("T3_cluster_x_education_chi2")
        add("T4_age_skeptic_vs_booster_cluster")
        return out

    if "trust_media" in v and "education" in v:
        add("T1_education_trust_media_kruskal")
        add("T1b_education_extremes_trust_media_mannwhitney")

    if "trust_police" in v and "trust_courts" in v:
        add("T2_trust_police_trust_courts_pearson")

    if "trust_media" in v and "trust_government" in v:
        add("T2b_trust_media_trust_government_pearson")

    if "trust_business" in v and "trust_church" in v:
        add("T6_trust_business_trust_church_pearson")

    if church_focus:
        if "age" in v:
            add("T5a_church_age")
        if "gender" in v:
            add("T5b_church_gender")
        if "education" in v:
            add("T5c_church_education")

    if not out:
        text_l = text.lower()
        keywords = {
            "T2_trust_police_trust_courts_pearson": ["полици", "суд", "правопоряд"],
            "T1_education_trust_media_kruskal": ["сми", "медиа", "образован"],
            "T3_cluster_x_education_chi2": ["кластер", "образован"],
            "T4_age_skeptic_vs_booster_cluster": ["возраст", "скептик", "кластер"],
            "T5a_church_age": ["церкв", "возраст"],
            "T5b_church_gender": ["церкв", "пол", "гендер"],
            "T5c_church_education": ["церкв", "образован"],
            "T6_trust_business_trust_church_pearson": ["бизнес", "церк", "trust_business"],
            "T2b_trust_media_trust_government_pearson": ["сми", "правительств", "медиа"],
        }
        for tid, kws in keywords.items():
            if tid in available_ids and any(kw in text_l for kw in kws):
                add(tid)

    return out


def run_statistical_tests(
    df_clean: pd.DataFrame,
    df_enc: pd.DataFrame,
    run_dir: Path,
    *,
    cluster_story: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Заполняет ``run_dir / "tests" /`` отчётами. Не требует LLM: опирается на те же столбцы, что и cluster_insights.
    """
    out = run_dir / "tests"
    out.mkdir(parents=True, exist_ok=True)
    battery: list[dict[str, Any]] = []

    # --- 1) Образование × trust_media (Kruskal-Wallis + попарно крайние группы) ---
    if "education" in df_clean.columns and "trust_media" in df_clean.columns:
        tmp = df_clean.assign(_tm=_numeric(df_clean["trust_media"])).dropna(subset=["_tm"])
        tmp = tmp[tmp["education"].notna()]
        groups = [g["_tm"].values for _, g in tmp.groupby(tmp["education"].astype(str)) if len(g) >= 10]
        if len(groups) >= 3:
            h_stat, p_kw = stats.kruskal(*groups)
            battery.append(
                {
                    "id": "T1_education_trust_media_kruskal",
                    "question": "Различается ли доверие к СМИ между группами образования?",
                    "test": "Kruskal–Wallis H",
                    "statistic": float(h_stat),
                    "pvalue": float(p_kw),
                    "n_groups": len(groups),
                    "group_sizes": [len(g) for g in groups],
                    "note": "Непараметрический аналог ANOVA по рангу; для порядковых шкал доверия уместнее, чем F-test без проверки нормальности.",
                }
            )
        # Попарно: две группы с max/min средним (если есть в story)
        grp_mean = tmp.groupby(tmp["education"].astype(str))["_tm"].mean()
        if len(grp_mean) >= 2:
            lo, hi = grp_mean.idxmin(), grp_mean.idxmax()
            a = tmp.loc[tmp["education"].astype(str) == lo, "_tm"].values
            b = tmp.loc[tmp["education"].astype(str) == hi, "_tm"].values
            if len(a) >= 10 and len(b) >= 10:
                mw = _mannwhitney_full(a, b)
                mw["id"] = "T1b_education_extremes_trust_media_mannwhitney"
                mw["question"] = f"Сравнение крайних групп по среднему доверию к СМИ: «{lo}» vs «{hi}»"
                mw["cohens_d"] = _cohens_d(a, b)
                battery.append(mw)

    # --- 2) Полиция × суды: корреляция + ДИ ---
    if "trust_police" in df_enc.columns and "trust_courts" in df_enc.columns:
        pair = df_enc[["trust_police", "trust_courts"]].dropna()
        if len(pair) >= 30:
            r = float(pair["trust_police"].corr(pair["trust_courts"]))
            t_r = r * np.sqrt((len(pair) - 2) / max(1e-9, 1 - r**2))
            p_r = 2 * (1 - stats.t.cdf(abs(t_r), len(pair) - 2))
            lo, hi = _pearson_r_ci(r, len(pair))
            battery.append(
                {
                    "id": "T2_trust_police_trust_courts_pearson",
                    "question": "Связаны ли доверие к полиции и к судам?",
                    "test": "Pearson correlation",
                    "r": round(r, 4),
                    "pvalue": float(p_r),
                    "n": len(pair),
                    "ci_r_95": [round(lo, 4), round(hi, 4)],
                    "interpretation": "r и ДИ по преобразованию Фишера; при |r|>0.2 и n большом эффект обычно не «шум».",
                }
            )

    # --- 3) Кластер × образование: chi-square + Cramér's V ---
    assign_path = run_dir / "eda" / "cluster_assignments.csv"
    if assign_path.is_file() and "education" in df_clean.columns:
        assign = pd.read_csv(assign_path, index_col=0)
        if "cluster_id" in assign.columns:
            merged = df_clean.join(assign[["cluster_id"]], how="inner")
            merged = merged[merged["education"].notna()]
            tab = pd.crosstab(merged["cluster_id"].astype(str), merged["education"].astype(str))
            if tab.shape[0] >= 2 and tab.shape[1] >= 2 and tab.values.sum() >= 40:
                chi = _chi2_cramers_v(tab)
                chi["id"] = "T3_cluster_x_education_chi2"
                chi["question"] = "Связан ли кластер (по профилю доверия) с уровнем образования?"
                battery.append(chi)

    # --- 4) Возраст: скептический кластер vs «усилитель» ---
    if cluster_story and "suggested_label_skeptic_cluster" in cluster_story and assign_path.is_file():
        sk = int(cluster_story["suggested_label_skeptic_cluster"])
        bo = int(cluster_story.get("suggested_label_booster_cluster", sk))
        assign = pd.read_csv(assign_path, index_col=0)
        if "cluster_id" in assign.columns and "age" in df_clean.columns:
            m = df_clean.join(assign[["cluster_id"]], how="inner")
            m["_age"] = _numeric(m["age"])
            a = m.loc[m["cluster_id"] == sk, "_age"].dropna().values
            b = m.loc[m["cluster_id"] == bo, "_age"].dropna().values
            if len(a) >= 15 and len(b) >= 15:
                wt = _welch_ttest_full(a, b)
                wt["id"] = "T4_age_skeptic_vs_booster_cluster"
                wt["question"] = f"Различается ли возраст между кластером-«скептиком» ({sk}) и кластером с высоким доверием ({bo})?"
                battery.append(wt)

    # --- 5) trust_church: корреляции с предикторами + ДИ ---
    if "trust_church" in df_enc.columns:
        for pred, pid in (
            ("age", "T5a_church_age"),
            ("gender", "T5b_church_gender"),
            ("education", "T5c_church_education"),
        ):
            if pred not in df_enc.columns:
                continue
            pair = df_enc[["trust_church", pred]].dropna()
            if len(pair) < 30:
                continue
            r = float(pair["trust_church"].corr(pair[pred]))
            lo, hi = _pearson_r_ci(r, len(pair))
            t_r = r * np.sqrt((len(pair) - 2) / max(1e-9, 1 - r**2))
            p_r = 2 * (1 - stats.t.cdf(abs(t_r), len(pair) - 2))
            battery.append(
                {
                    "id": pid,
                    "question": f"Насколько {pred} связан с доверием к церкви (линейно, по Пирсону)?",
                    "test": "Pearson correlation",
                    "predictor": pred,
                    "r": round(r, 4),
                    "pvalue": float(p_r),
                    "n": len(pair),
                    "ci_r_95": [round(lo, 4), round(hi, 4)],
                }
            )

    # --- 6) Доп. пары «доверие × доверие» (часто в гипотезах LLM, см. cluster_story focus_pairs) ---
    extra_pairs: tuple[tuple[str, str, str, str], ...] = (
        (
            "trust_media",
            "trust_government",
            "T2b_trust_media_trust_government_pearson",
            "Связаны ли доверие к СМИ и доверие к правительству?",
        ),
        (
            "trust_business",
            "trust_church",
            "T6_trust_business_trust_church_pearson",
            "Связаны ли доверие к крупному бизнесу и доверие к церкви?",
        ),
    )
    for ca, cb, tid, q in extra_pairs:
        row = _pearson_pair_row(df_enc, ca, cb, tid, q)
        if row:
            battery.append(row)

    # --- Сводка «не шум» ---
    sig = [b for b in battery if isinstance(b.get("pvalue"), (int, float)) and b["pvalue"] < 0.05]
    summary = {
        "n_tests_reported": len(battery),
        "n_significant_p_lt_0_05": len(sig),
        "tests_with_ids": [b.get("id") for b in battery if b.get("id")],
        "note": "Интерпретируй совместно с множественными сравнениями; при 10+ тестах возможны ложные открытия (рассмотри FDR).",
    }

    report = {"battery": battery, "summary": summary}
    (out / "statistical_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    md_lines = [
        "# Статистические проверки (автоматическая батарея)",
        "",
        f"Всего тестов: {len(battery)}; с p < 0.05: {len(sig)}.",
        "",
        summary["note"],
        "",
    ]
    for b in battery:
        bid = b.get("id", "?")
        md_lines.append(f"## {bid}")
        md_lines.append("")
        q = b.get("question", "")
        if q:
            md_lines.append(f"*{q}*")
            md_lines.append("")
        for k, v in b.items():
            if k in ("id", "question"):
                continue
            md_lines.append(f"- **{k}:** {v}")
        md_lines.append("")

    (out / "statistical_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Сопоставление гипотез с тестами (variables_involved + формулировка)
    hyp_path = run_dir / "hypotheses" / "hypotheses.json"
    if hyp_path.is_file():
        try:
            hyp_data = json.loads(hyp_path.read_text(encoding="utf-8"))
            hyps = hyp_data.get("hypotheses") or []
            available_ids = {str(b.get("id")) for b in battery if b.get("id")}
            links: list[dict[str, Any]] = []
            for i, h in enumerate(hyps):
                if not isinstance(h, dict):
                    continue
                matched = link_hypothesis_to_battery_ids(h, available_ids)
                if matched:
                    links.append({"hypothesis_index": i, "title": h.get("title"), "related_tests": matched})
            (out / "hypothesis_to_test_map.json").write_text(
                json.dumps({"links": links}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("hypothesis map: %s", e)

    return report

