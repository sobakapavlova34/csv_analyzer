from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from survey_agent.reporting.report_builder import build_html_report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, help="Каталог прогона (с preprocess/, eda/, …)")
    p.add_argument("--latest", action="store_true", help="Взять последний подкаталог в artifacts/")
    args = p.parse_args()

    if args.latest:
        root = _ROOT / "artifacts"
        if not root.is_dir():
            print("Нет папки artifacts/", file=sys.stderr)
            sys.exit(1)
        dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
        if not dirs:
            print("Пусто в artifacts/", file=sys.stderr)
            sys.exit(1)
        run_dir = dirs[0]
    elif args.run_dir:
        run_dir = args.run_dir
    else:
        p.print_help()
        sys.exit(1)

    hyp_json = run_dir / "hypotheses" / "hypotheses.json"
    print("hypotheses.json:", hyp_json.resolve(), "| exists:", hyp_json.is_file())
    out = build_html_report(run_dir)
    print("Written:", out.resolve())
    if hyp_json.is_file():
        import json as _json

        d = _json.loads(hyp_json.read_text(encoding="utf-8"))
        hs = d.get("hypotheses") or []
        print("Loaded:", len(hs) if isinstance(hs, list) else "?", "hypotheses,", "source=", d.get("source"))


if __name__ == "__main__":
    main()
