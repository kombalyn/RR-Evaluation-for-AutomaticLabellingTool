#!/usr/bin/env python3
"""
RR értékelés CLI: légzési jel (1. oszlop) vs. referencia (utolsó oszlop) — MAE, RMSE, bootstrap CI.

Példa:
  cd RR_eval
  python run_eval.py --input path/to/wave.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import RREvalConfig
from evaluation import evaluate_signal_vs_reference
from io_wave import load_wave_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RR becslés vs. referencia értékelés")
    p.add_argument("--input", type=Path, required=True, help="CSV: 1. oszlop jel, utolsó referencia")
    p.add_argument("--delimiter", type=str, default=",")
    p.add_argument("--signal-col", type=int, default=0)
    p.add_argument("--reference-col", type=int, default=-1)
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--window-sec", type=float, default=20.0)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    p.add_argument("--json-out", type=Path, default=None, help="Opcionális összefoglaló JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        print(f"Nem található: {args.input}", file=sys.stderr)
        sys.exit(1)

    signal, reference = load_wave_matrix(
        args.input,
        delimiter=args.delimiter,
        signal_col=args.signal_col,
        reference_col=args.reference_col,
    )

    cfg = RREvalConfig(
        fps=args.fps,
        window_seconds=args.window_sec,
        bootstrap_n=args.bootstrap_n,
    )

    res = evaluate_signal_vs_reference(signal, reference, cfg)

    summary = {
        "input": str(args.input.resolve()),
        "fps": cfg.fps,
        "window_seconds": cfg.window_seconds,
        "n_windows_total": res.n_windows_total,
        "n_windows_valid": res.n_windows_valid,
        "coverage": res.coverage,
        "MAE": res.mae,
        "RMSE": res.rmse,
        "MAE_bootstrap_95CI": [res.mae_ci_lower, res.mae_ci_upper],
    }

    print(json.dumps(summary, indent=2))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Mentve: {args.json_out}")


if __name__ == "__main__":
    main()
