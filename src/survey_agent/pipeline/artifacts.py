from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def new_run_dir(base: Path | None = None) -> Path:
    root = base or Path("artifacts")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run = root / stamp
    return run


def ensure_stage_dirs(run_dir: Path) -> dict[str, Path]:
    d = {
        "root": run_dir,
        "preprocess": run_dir / "preprocess",
        "eda": run_dir / "eda",
        "hypotheses": run_dir / "hypotheses",
        "tests": run_dir / "tests",
        "logs": run_dir / "logs",
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d
