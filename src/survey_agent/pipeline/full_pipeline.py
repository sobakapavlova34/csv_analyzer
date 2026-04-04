"""
Полный прогон: предобработка → EDA → гипотезы; всё складывается в ``artifacts/<run_id>/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from survey_agent.data_schema import (
    MIN_TRUST_COLUMNS_FOR_CORE_EDA,
    TRUST_COLUMN_PREFIX,
    structure_hint,
    trust_column_names,
)
from survey_agent.pipeline.artifacts import ensure_stage_dirs, new_run_dir
from survey_agent.pipeline.eda import run_eda
from survey_agent.pipeline.hypotheses import load_config_optional, run_hypotheses_stage
from survey_agent.pipeline.hypothesis_tests import run_statistical_tests
from survey_agent.pipeline.preprocess import run_preprocess
from survey_agent.reporting.report_builder import build_html_report

logger = logging.getLogger(__name__)


def _read_csv_flexible(path: Path, encoding: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        if encoding.lower() in ("utf-8", "utf8"):
            return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        raise


def run_full_pipeline(
    csv_path: str | Path,
    *,
    artifact_root: Path | None = None,
    encoding: str = "utf-8",
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Возвращает словарь с путями и кратким summary; пишет ``run_manifest.json`` в каталог прогона.

    Вход: любой путь к ``.csv`` с похожей структурой (см. ``survey_agent.data_schema``), не привязка к одному файлу.
    """
    csv_path = Path(csv_path).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV не найден: {csv_path}")

    run_dir = new_run_dir(artifact_root)
    dirs = ensure_stage_dirs(run_dir)

    raw = _read_csv_flexible(csv_path, encoding)
    n_trust = len(trust_column_names(raw))
    if n_trust < MIN_TRUST_COLUMNS_FOR_CORE_EDA:
        logger.warning(
            "Столбцов %s*: %s (нужно ≥%s для кластеров/корреляций). %s",
            TRUST_COLUMN_PREFIX,
            n_trust,
            MIN_TRUST_COLUMNS_FOR_CORE_EDA,
            structure_hint(),
        )

    df_clean, df_enc, prep_report = run_preprocess(raw, dirs["preprocess"], random_seed=random_seed)

    eda_report = run_eda(
        df_clean,
        df_enc,
        prep_report,
        dirs["eda"],
        csv_path_for_state=csv_path,
        random_seed=random_seed,
    )

    cfg = load_config_optional()
    hyp_report = run_hypotheses_stage(run_dir, cfg)

    cluster_story_path = dirs["eda"] / "cluster_story.json"
    cluster_story: dict[str, Any] = {}
    if cluster_story_path.is_file():
        cluster_story = json.loads(cluster_story_path.read_text(encoding="utf-8"))

    test_report = run_statistical_tests(
        df_clean,
        df_enc,
        run_dir,
        cluster_story=cluster_story if cluster_story else None,
    )

    report_html: Path | None = None
    try:
        report_html = build_html_report(run_dir)
    except Exception as e:
        logger.warning("HTML report: %s", e)

    manifest = {
        "run_dir": str(run_dir),
        "csv_source": str(csv_path),
        "preprocess": {
            "dir": str(dirs["preprocess"]),
            "files": ["report.json", "clean.csv", "encoded_features.csv", "column_roles.json"],
        },
        "eda": {
            "dir": str(dirs["eda"]),
            "files": [
                "distributions.json",
                "correlation_pearson.csv",
                "correlation_tool.json",
                "clusters_kmeans.json",
                "cluster_story.json",
                "cluster_profiles.json",
                "cluster_assignments.csv",
                "pca_projection.json",
                "pca_coordinates_sample.csv",
                "eda_report.json",
                "eda_summary.md",
            ],
        },
        "hypotheses": {
            "dir": str(dirs["hypotheses"]),
            "source": hyp_report.get("source"),
            "n_hypotheses": len(hyp_report.get("hypotheses") or []),
        },
        "tests": {
            "dir": str(dirs["tests"]),
            "n_tests": test_report.get("summary", {}).get("n_tests_reported"),
            "files": ["statistical_report.json", "statistical_report.md", "hypothesis_to_test_map.json"],
        },
        "report": {
            "html": str(report_html) if report_html else None,
            "dir": str(run_dir / "report"),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dirs["logs"] / "pipeline.log").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest
