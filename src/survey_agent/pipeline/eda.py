"""
EDA анализ: распределения, корреляции, k-means, PCA.
Результаты — отдельные файлы в ``artifacts/.../eda/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from survey_agent.data_schema import trust_column_names
from survey_agent.pipeline.cluster_insights import run_cluster_insights
from survey_agent.state import SessionState
from survey_agent.tools.runner import run_tool
from survey_agent.types import ToolCall

logger = logging.getLogger(__name__)


def _distributions(df_clean: pd.DataFrame, max_cats: int = 15) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in df_clean.columns:
        s = df_clean[c]
        try:
            if pd.api.types.is_numeric_dtype(s):
                x = pd.to_numeric(s, errors="coerce")
                out[c] = {
                    "type": "numeric",
                    "missing_share": float(x.isna().mean()),
                    "mean": float(np.nanmean(x)) if x.notna().any() else None,
                    "std": float(np.nanstd(x)) if x.notna().any() else None,
                    "min": float(np.nanmin(x)) if x.notna().any() else None,
                    "max": float(np.nanmax(x)) if x.notna().any() else None,
                    "quantiles": {
                        "q25": float(np.nanquantile(x, 0.25)),
                        "q50": float(np.nanquantile(x, 0.50)),
                        "q75": float(np.nanquantile(x, 0.75)),
                    }
                    if x.notna().sum() > 0
                    else {},
                }
            else:
                vc = s.astype(str).value_counts().head(max_cats)
                out[c] = {
                    "type": "categorical",
                    "n_unique": int(s.nunique(dropna=True)),
                    "missing_share": float(s.isna().mean()),
                    "top_values": {str(k): int(v) for k, v in vc.items()},
                }
        except Exception as e:
            out[c] = {"error": str(e)}
    return out


def run_eda(
    df_clean: pd.DataFrame,
    df_enc: pd.DataFrame,
    preprocess_report: dict[str, Any],
    out_dir: Path,
    csv_path_for_state: Path,
    *,
    random_seed: int = 42,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = SessionState(csv_path=csv_path_for_state, random_seed=random_seed)
    state.df = df_enc.copy()

    summary: dict[str, Any] = {"stages": []}

    # 1) Распределения
    dist = _distributions(df_clean)
    (out_dir / "distributions.json").write_text(
        json.dumps(dist, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    summary["stages"].append({"name": "distributions", "file": "distributions.json"})

    # 2) Корреляции (по числовому encoded; при необходимости сузить trust_* + age + weight)
    trust_like = trust_column_names(df_enc)
    age_weight = [c for c in ("age", "weight") if c in df_enc.columns]
    corr_cols = trust_like + age_weight
    if len(corr_cols) < 2:
        corr_cols = list(df_enc.columns)[: min(30, len(df_enc.columns))]

    r_corr = run_tool(
        state,
        ToolCall("numeric_correlation", {"columns": corr_cols, "min_non_null_share": 0.4}),
    )
    corr_payload = r_corr.payload if r_corr.ok else {"error": r_corr.error}
    (out_dir / "correlation_tool.json").write_text(
        json.dumps(corr_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if r_corr.ok and "correlation_matrix" in r_corr.payload:
        cm = r_corr.payload["correlation_matrix"]
        if isinstance(cm, dict):
            pd.DataFrame(cm).to_csv(out_dir / "correlation_pearson.csv", encoding="utf-8")
    summary["stages"].append({"name": "correlation", "file": "correlation_pearson.csv"})

    # 3) K-means по trust_*
    kmeans_payload: dict[str, Any] = {}
    if len(trust_like) >= 2:
        r_k = run_tool(
            state,
            ToolCall(
                "kmeans_cluster_summary",
                {"feature_columns": trust_like, "n_clusters": 4, "standardize": True},
            ),
        )
        kmeans_payload = r_k.to_message_dict() if r_k else {}
        (out_dir / "clusters_kmeans.json").write_text(
            json.dumps(kmeans_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        summary["stages"].append({"name": "kmeans", "file": "clusters_kmeans.json"})
    else:
        (out_dir / "clusters_kmeans.json").write_text("{}", encoding="utf-8")

    # 4) PCA 2D
    pca_cols = trust_like if len(trust_like) >= 2 else corr_cols[: min(10, len(corr_cols))]
    pca_payload: dict[str, Any] = {}
    if len(pca_cols) >= 2:
        r_p = run_tool(
            state,
            ToolCall(
                "pca_projection_2d",
                {"feature_columns": pca_cols, "sample_rows": 800, "standardize": True},
            ),
        )
        pca_payload = r_p.to_message_dict() if r_p else {}
        (out_dir / "pca_projection.json").write_text(
            json.dumps(pca_payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if r_p.ok and "coordinates_sample" in r_p.payload:
            pd.DataFrame(r_p.payload["coordinates_sample"]).to_csv(
                out_dir / "pca_coordinates_sample.csv", index=False, encoding="utf-8"
            )
        summary["stages"].append({"name": "pca", "file": "pca_projection.json"})

    md_lines = [
        "# Разведочный анализ",
        "",
        f"- Строк: {len(df_clean)}, столбцов: {len(df_clean.columns)}",
        f"- Корреляция: см. `correlation_pearson.csv`, детали `correlation_tool.json`",
        f"- Кластеры (trust): см. `clusters_kmeans.json`",
        f"- **Профили кластеров и сюжеты для гипотез:** `cluster_story.json`, `cluster_profiles.json`",
        f"- PCA: см. `pca_projection.json`, точки `pca_coordinates_sample.csv`",
        "",
    ]
    (out_dir / "eda_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    # 5) Кластеры + соцдем + «сюжеты» для содержательных гипотез
    cluster_story: dict[str, Any] = {}
    if len(trust_like) >= 2:
        try:
            cluster_story = run_cluster_insights(
                df_clean,
                df_enc,
                out_dir,
                n_clusters=4,
                random_seed=random_seed,
            )
            summary["stages"].append(
                {"name": "cluster_insights", "files": ["cluster_story.json", "cluster_profiles.json"]}
            )
        except Exception as e:
            logger.warning("cluster_insights failed: %s", e)
            cluster_story = {"error": str(e)}

    eda_report = {
        "summary": summary,
        "preprocess_n_columns": preprocess_report.get("n_columns"),
        "trust_columns_used": trust_like,
        "cluster_story_keys": list(cluster_story.keys()) if cluster_story else [],
    }
    (out_dir / "eda_report.json").write_text(
        json.dumps(eda_report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    log_path = out_dir.parent / "logs" / "eda.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    return eda_report
