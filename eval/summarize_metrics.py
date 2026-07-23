#!/usr/bin/env python3
"""Summarize per-sample PIE metric CSV into category means."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EVAL_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute mean values for each PIE metric column.")
    parser.add_argument(
        "--input",
        type=Path,
        default=EVAL_ROOT / "evaluation_result.csv",
        help="Input CSV produced by eval/evaluate.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_ROOT / "metrics_summary.csv",
        help="CSV path to save one-row mean summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Metric CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    metric_df = df.drop(columns=["file_id"], errors="ignore")
    metric_df = metric_df.apply(pd.to_numeric, errors="coerce")
    mean_values = metric_df.mean(skipna=True)

    print(f"Summary for: {args.input.resolve()}")
    print("-" * 60)
    for metric, mean in mean_values.items():
        valid_count = int(metric_df[metric].count())
        print(f"{metric}: {mean:.6f} (valid={valid_count})")

    summary = mean_values.to_frame().T
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print("-" * 60)
    print(f"Saved summary to {args.output.resolve()}")


if __name__ == "__main__":
    main()
