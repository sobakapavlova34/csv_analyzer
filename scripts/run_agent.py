#!/usr/bin/env python3
"""
Запуск LLM-агента (BotHub / OpenAI-совместимый API).

  export BOTHUB_API_KEY=...
  # опционально: BOTHUB_BASE_URL, BOTHUB_MODEL, AGENT_MAX_STEPS
  PYTHONPATH=src python scripts/run_agent.py --csv /путь/к/данным.csv
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
from survey_agent.config import AgentConfig
from survey_agent.agent.orchestrator import run_from_csv


def main() -> None:
    p = argparse.ArgumentParser(
        description="Survey analysis agent (Bothub LLM + tools)",
        epilog=structure_hint(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, required=True, help="Входной .csv (любой путь)")
    p.add_argument("--encoding", default="utf-8", help="Кодировка CSV")
    p.add_argument("--transcript", type=Path, help="Сохранить JSON-транскрипт шагов")
    args = p.parse_args()

    cfg = AgentConfig.from_env()
    state, transcript = run_from_csv(args.csv, cfg=cfg, encoding=args.encoding)
    print("phase:", state.phase)
    print("tool_log entries:", len(state.tool_log))
    print("--- transcript ---")
    print(json.dumps(transcript, ensure_ascii=False, indent=2)[:8000])
    if args.transcript:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
        with open(args.transcript, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        print("wrote", args.transcript)


if __name__ == "__main__":
    main()
