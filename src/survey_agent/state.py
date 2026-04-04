from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SessionState:
    csv_path: Path
    encoding: str = "utf-8"
    random_seed: int = 42

    df: pd.DataFrame | None = None

    column_kind: dict[str, str] = field(default_factory=dict)
    phase: str = "init"
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    test_results: list[dict[str, Any]] = field(default_factory=list)
    tool_log: list[dict[str, Any]] = field(default_factory=list)

    artifact_dir: Path | None = None

    def ensure_df(self) -> pd.DataFrame:
        if self.df is None:
            self.df = pd.read_csv(self.csv_path, encoding=self.encoding, low_memory=False)
        return self.df
