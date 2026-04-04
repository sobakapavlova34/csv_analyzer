#!/usr/bin/env python3
"""
CLI: один CSV — смешанные типы, по умолчанию 5000×174.

Usage:
  PYTHONPATH=src python scripts/generate_dataset.py --out ./survey_synthetic.csv
  PYTHONPATH=src python scripts/generate_dataset.py --rows 500 --cols 50 --out /tmp/small.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running without install: repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from survey_synthetic.generator import generate_survey_dataframe
from survey_synthetic.schema import DEFAULT_N_COLS, DEFAULT_N_ROWS, DEFAULT_SEED


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic survey microdata (ESS/WVS-like).")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("survey_synthetic.csv"),
        help="Куда сохранить CSV (произвольный путь)",
    )
    p.add_argument("--rows", type=int, default=DEFAULT_N_ROWS, help="Number of respondents")
    p.add_argument("--cols", type=int, default=DEFAULT_N_COLS, help="Total columns (core + q_001…)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    p.add_argument("--missing-rate", type=float, default=0.04, help="NaN injection rate in trust cols")
    p.add_argument("--empty-cat-rate", type=float, default=0.015, help="Empty string rate in categories")
    p.add_argument("--mixed-trust-rate", type=float, default=0.02, help="String/junk rate in trust cols")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = generate_survey_dataframe(
        n_rows=args.rows,
        seed=args.seed,
        n_cols=args.cols,
        missing_rate=args.missing_rate,
        empty_cat_rate=args.empty_cat_rate,
        mixed_trust_rate=args.mixed_trust_rate,
    )
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows × {len(df.columns)} cols → {args.out}")


if __name__ == "__main__":
    main()
