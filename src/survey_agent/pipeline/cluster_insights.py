"""
Углублённый кластерный срез + соцдем-профили + «сюжеты» для гипотез.

Цель — дать LLM конкретику вместо банального «кластеры различаются».
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from survey_agent.data_schema import TRUST_COLUMN_PREFIX, trust_column_names

logger = logging.getLogger(__name__)

SOCIO_CANDIDATES = ("education", "gender", "income_subj", "settlement", "region_id")
TRUST_PAIR_FOCUS = (
    ("trust_police", "trust_courts"),
    ("trust_media", "trust_government"),
    ("trust_church", "trust_army"),
)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def run_cluster_insights(
    df_clean: pd.DataFrame,
    df_enc: pd.DataFrame,
    out_dir: Path,
    *,
    n_clusters: int = 4,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Пишет в ``out_dir``:
    - ``cluster_profiles.json`` — по каждому кластеру: средние по trust, возраст, распределение образования/пола
    - ``cluster_story.json`` — кто «тотальные скептики», топ корреляций между институтами, сравнение силы предикторов для trust_church
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trust_cols = trust_column_names(df_enc)
    if len(trust_cols) < 2:
        empty = {"error": f"insufficient {TRUST_COLUMN_PREFIX}* columns"}
        (out_dir / "cluster_story.json").write_text(json.dumps(empty), encoding="utf-8")
        return empty

    mask = df_enc[trust_cols].notna().all(axis=1)
    if mask.sum() < n_clusters * 10:
        mask = df_enc[trust_cols].notna().any(axis=1)

    sub_enc = df_enc.loc[mask].copy()
    sub_clean = df_clean.loc[mask].copy()
    X = sub_enc[trust_cols].values.astype(float)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = km.fit_predict(Xs)
    sub_clean["_cluster"] = labels

    # Средний «уровень доверия» по кластеру (для поиска «тотальных скептиков»)
    cluster_mean: dict[int, float] = {}
    for cid in sorted(sub_clean["_cluster"].unique()):
        idx = sub_clean.index[sub_clean["_cluster"] == cid]
        cluster_mean[int(cid)] = float(sub_enc.loc[idx, trust_cols].mean().mean())
    skeptic_cluster = min(cluster_mean, key=lambda k: cluster_mean[k])
    booster_cluster = max(cluster_mean, key=lambda k: cluster_mean[k])

    profiles: dict[str, Any] = {}
    for cid in sorted(sub_clean["_cluster"].unique()):
        g = sub_clean[sub_clean["_cluster"] == cid]
        ge = sub_enc.loc[g.index, trust_cols]
        block: dict[str, Any] = {
            "n": int(len(g)),
            "mean_trust_per_institution": {c: float(ge[c].mean()) for c in trust_cols},
            "mean_trust_overall": float(ge.mean().mean()),
        }
        if "age" in g.columns:
            a = _numeric_series(g, "age")
            if a.notna().any():
                block["age_mean"] = float(a.mean())
                block["age_median"] = float(a.median())
        for col in SOCIO_CANDIDATES:
            if col in g.columns and g[col].notna().any():
                vc = g[col].astype(str).value_counts(normalize=True).head(6)
                block[f"share_{col}"] = {str(k): round(float(v), 4) for k, v in vc.items()}
        profiles[str(cid)] = block

    # Пары институтов (Пирсон по полным строкам)
    cm = sub_enc[trust_cols].corr()
    pair_rows: list[dict[str, Any]] = []
    for i, a in enumerate(trust_cols):
        for b in trust_cols[i + 1 :]:
            r = float(cm.loc[a, b]) if a in cm.index and b in cm.columns else None
            if r is not None and r == r:
                pair_rows.append({"a": a, "b": b, "r": round(r, 4)})
    pair_rows.sort(key=lambda x: -abs(x["r"]))
    top_pairs = pair_rows[:12]

    focus_pairs: list[dict[str, Any]] = []
    for a, b in TRUST_PAIR_FOCUS:
        if a in cm.index and b in cm.columns:
            focus_pairs.append(
                {"a": a, "b": b, "r": round(float(cm.loc[a, b]), 4), "label": "фокус (полиция/суды, СМИ/власть, …)"}
            )

    # Сила предикторов для trust_church: |корреляция Пирсона| с age, gender, education (encoded)
    church = "trust_church"
    predictor_rank: list[dict[str, Any]] = []
    if church in sub_enc.columns:
        cand = [c for c in ("age", "gender", "education", "income_subj", "settlement") if c in sub_enc.columns]
        for p in cand:
            pair = sub_enc[[church, p]].dropna()
            if len(pair) < 30:
                continue
            r = float(pair[church].corr(pair[p]))
            predictor_rank.append({"predictor": p, "r_pearson": round(r, 4), "abs_r": round(abs(r), 4)})
        predictor_rank.sort(key=lambda x: -x["abs_r"])

    # Образование × доверие к СМИ (средние по группам)
    edu_media: dict[str, Any] = {}
    if "education" in sub_clean.columns and "trust_media" in sub_clean.columns:
        tm = _numeric_series(sub_clean, "trust_media")
        tmp = sub_clean.assign(_tm=tm).dropna(subset=["_tm"])
        if len(tmp) > 20:
            grp = tmp.groupby(tmp["education"].astype(str), dropna=False)["_tm"].agg(["mean", "count"])
            grp = grp[grp["count"] >= 15].sort_values("mean")
            edu_media = {
                "lowest_trust_media_group": str(grp.index[0]) if len(grp) else None,
                "highest_trust_media_group": str(grp.index[-1]) if len(grp) else None,
                "group_means_trust_media": {str(i): float(r["mean"]) for i, r in grp.iterrows()},
            }

    story: dict[str, Any] = {
        "n_rows_used": int(len(sub_clean)),
        "n_clusters": n_clusters,
        "cluster_mean_trust_overall": {str(k): float(v) for k, v in cluster_mean.items()},
        "suggested_label_skeptic_cluster": skeptic_cluster,
        "suggested_label_booster_cluster": booster_cluster,
        "cluster_profiles": profiles,
        "top_trust_correlations": top_pairs,
        "focus_pairs": focus_pairs,
        "predictor_strength_for_trust_church": predictor_rank,
        "education_vs_trust_media": edu_media,
        "hints_for_hypotheses": [
            "Сравни профили кластеров {0} (скептики) и {1} (высокое доверие) по образованию и возрасту.".format(
                skeptic_cluster, booster_cluster
            ),
            "Используй top_trust_correlations: если полиция и суды сильно коррелируют — формулируй гипотезу про «связку» институтов правопорядка.",
            "Если age в predictor_strength_for_trust_church выше gender/education — гипотеза про возраст как главный фактор доверия к церкви.",
            "Если в education_vs_trust_media есть явный разрыв между «высшее» и другими — гипотеза в стиле ТЗ про СМИ и образование.",
        ],
    }

    assign = pd.DataFrame({"cluster_id": sub_clean["_cluster"].values}, index=sub_clean.index)
    assign.to_csv(out_dir / "cluster_assignments.csv", encoding="utf-8")
    story["cluster_assignments_csv"] = "cluster_assignments.csv"

    (out_dir / "cluster_profiles.json").write_text(
        json.dumps({"profiles": profiles, "kmeans_inertia": float(km.inertia_)}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "cluster_story.json").write_text(
        json.dumps(story, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return story
