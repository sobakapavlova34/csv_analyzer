#!/usr/bin/env python3
"""
Полный цикл: предобработка → EDA → гипотезы (LLM при BOTHUB_API_KEY).

  PYTHONPATH=src python scripts/run_pipeline.py --csv /путь/к/любому.csv

Артефакты: artifacts/<timestamp>/preprocess|eda|hypotheses|logs/
Вход — любой CSV с похожей структурой (см. survey_agent.data_schema).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from survey_agent.data_schema import structure_hint
from survey_agent.pipeline.full_pipeline import run_full_pipeline


def main() -> None:
    p = argparse.ArgumentParser(
        description="Пайплайн анализа опроса: любой CSV с подходящими столбцами.",
        epilog=structure_hint(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Путь к входному .csv (имя файла произвольное)",
    )
    p.add_argument(
        "--encoding",
        default="utf-8",
        help="Кодировка CSV (при ошибке для utf-8 будет пробован utf-8-sig)",
    )
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = p.parse_args()

    m = run_full_pipeline(args.csv, artifact_root=args.artifacts, encoding=args.encoding)
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("\nRun directory:", m["run_dir"])


if __name__ == "__main__":
    main()
