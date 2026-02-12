#!/usr/bin/env python3
"""Create a stratified subset of the IT tickets CSV.

Default behavior:
- input:  data/raw/IT Support Ticket Data.csv
- output: data/processed/IT Support Ticket Data.stratified_3000.csv
- size:   3000 rows
- stratify on Department + Priority (joint label)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified dataset subset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/IT Support Ticket Data.csv"),
        help="Path to full input CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/IT Support Ticket Data.stratified_3000.csv"),
        help="Path to save subset CSV",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=3000,
        help="Number of rows in subset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    if args.n_rows <= 0 or args.n_rows > len(df):
        raise ValueError(f"--n-rows must be in [1, {len(df)}], got {args.n_rows}")

    required_cols = ["Department", "Priority"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for stratification: {missing}")

    strat_series = (
        df["Department"].astype(str).str.strip()
        + "||"
        + df["Priority"].astype(str).str.strip()
    )

    sampled, _ = train_test_split(
        df,
        train_size=args.n_rows,
        random_state=args.seed,
        stratify=strat_series,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled = sampled.reset_index(drop=True)
    sampled.to_csv(args.output, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(sampled)}")
    print(f"Saved subset: {args.output}")
    print("\nDepartment distribution (%):")
    print((sampled["Department"].value_counts(normalize=True) * 100).round(2).to_string())
    print("\nPriority distribution (%):")
    print((sampled["Priority"].value_counts(normalize=True) * 100).round(2).to_string())


if __name__ == "__main__":
    main()
