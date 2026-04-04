"""Графики для HTML-отчёта (matplotlib, backend Agg)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# Кэш matplotlib в проекте (удобно в CI / песочницах без записи в ~)
_mpl_dir = Path(__file__).resolve().parents[3] / ".mpl_cache"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from survey_agent.data_schema import trust_column_names

logger = logging.getLogger(__name__)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        logger.warning("read %s: %s", path, e)
        return None


def figure_correlation_heatmap(corr_csv: Path, out_path: Path, max_cols: int = 12) -> bool:
    if not corr_csv.is_file():
        return False
    try:
        df = pd.read_csv(corr_csv, index_col=0, encoding="utf-8")
    except Exception:
        return False
    if df is None or df.empty:
        return False
    num = df.apply(pd.to_numeric, errors="coerce")
    cols = trust_column_names(num)[:max_cols]
    if len(cols) < 2:
        cols = list(num.columns)[: min(max_cols, len(num.columns))]
    if len(cols) < 2:
        return False
    sub = num[cols].astype(float)
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.5), max(6, len(cols) * 0.45)))
    im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Пирсон r")
    ax.set_title("Корреляции между шкалами доверия")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_education_trust_media(clean_csv: Path, out_path: Path) -> bool:
    df = _safe_read_csv(clean_csv)
    if df is None or "education" not in df.columns or "trust_media" not in df.columns:
        return False
    df = df.copy()
    df["trust_media"] = pd.to_numeric(df["trust_media"], errors="coerce")
    df = df.dropna(subset=["trust_media"])
    vc = df["education"].astype(str).value_counts().head(8).index
    sub = df[df["education"].astype(str).isin(vc)]
    if len(sub) < 30:
        return False
    order = sub.groupby(sub["education"].astype(str))["trust_media"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [sub.loc[sub["education"].astype(str) == g, "trust_media"].values for g in order]
    ax.boxplot(data, labels=[str(g)[:22] for g in order], showmeans=True)
    ax.set_ylabel("Доверие к СМИ (шкала)")
    ax.set_xlabel("Образование")
    ax.set_title("Распределение доверия к СМИ по группам образования")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_pca(pca_csv: Path, out_path: Path) -> bool:
    df = _safe_read_csv(pca_csv)
    if df is None or len(df) < 20:
        return False
    if "x" not in df.columns or "y" not in df.columns:
        return False
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["x"], df["y"], alpha=0.35, s=12, c="#2563eb")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA по шкалам доверия (проекция на плоскость)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_cluster_bar(cluster_story: dict[str, Any], out_path: Path) -> bool:
    cm = cluster_story.get("cluster_mean_trust_overall")
    if not cm:
        return False
    keys = sorted(cm.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
    vals = [float(cm[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([f"Кластер {k}" for k in keys], vals, color="#059669")
    ax.set_ylabel("Среднее доверие (по всем институтам)")
    ax.set_title("Средний уровень доверия по кластерам")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def build_all_figures(run_dir: Path, cluster_story: dict[str, Any] | None) -> dict[str, str]:
    """Возвращает {имя: относительный путь от report/index.html}."""
    fig_dir = run_dir / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel: dict[str, str] = {}
    eda = run_dir / "eda"
    pre = run_dir / "preprocess"

    p = fig_dir / "corr_trust.png"
    if figure_correlation_heatmap(eda / "correlation_pearson.csv", p):
        rel["corr"] = "figures/corr_trust.png"

    p = fig_dir / "edu_media.png"
    if figure_education_trust_media(pre / "clean.csv", p):
        rel["edu_media"] = "figures/edu_media.png"

    p = fig_dir / "pca.png"
    if figure_pca(eda / "pca_coordinates_sample.csv", p):
        rel["pca"] = "figures/pca.png"

    if cluster_story:
        p = fig_dir / "clusters.png"
        if figure_cluster_bar(cluster_story, p):
            rel["clusters"] = "figures/clusters.png"

    return rel
