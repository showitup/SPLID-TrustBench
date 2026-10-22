#!/usr/bin/env python3
"""Compare public prefix outputs under original and perturbed future suffixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["object_id", "time_idx", "direction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--perturbed", type=Path, required=True)
    parser.add_argument("--cutoffs", type=Path, required=True)
    parser.add_argument("--columns", nargs="*")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def audit(
    reference: pd.DataFrame,
    perturbed: pd.DataFrame,
    cutoffs: pd.DataFrame,
    columns: list[str] | None,
    tolerance: float,
) -> dict:
    required_cutoff = {"object_id", "direction", "cut_time_idx"}
    missing_cutoff = sorted(required_cutoff.difference(cutoffs.columns))
    if missing_cutoff:
        return {"passed": False, "errors": [f"cutoffs missing columns: {missing_cutoff}"]}
    if not np.isfinite(tolerance) or tolerance < 0:
        return {"passed": False, "errors": ["tolerance must be finite and nonnegative"]}
    if cutoffs.duplicated(["object_id", "direction"]).any():
        return {"passed": False, "errors": ["cutoffs contain duplicate object/direction keys"]}
    for name, frame in (("reference", reference), ("perturbed", perturbed)):
        missing = sorted(set(KEYS).difference(frame.columns))
        if missing:
            return {"passed": False, "errors": [f"{name} missing columns: {missing}"]}
        if frame.duplicated(KEYS).any():
            return {"passed": False, "errors": [f"{name} contains duplicate keys"]}

    if columns is None:
        excluded = set(KEYS) | {"object_id", "time_idx", "direction"}
        columns = sorted(
            column for column in set(reference.columns).intersection(perturbed.columns)
            if column not in excluded
            and pd.api.types.is_numeric_dtype(reference[column])
            and pd.api.types.is_numeric_dtype(perturbed[column])
        )
    missing_columns = sorted(
        set(columns).difference(reference.columns).union(set(columns).difference(perturbed.columns))
    )
    if missing_columns:
        return {"passed": False, "errors": [f"comparison columns missing: {missing_columns}"]}
    if not columns:
        return {"passed": False, "errors": ["no numeric comparison columns"]}

    cutoff_map = cutoffs.set_index(["object_id", "direction"])["cut_time_idx"].to_dict()
    reference = reference.copy()
    perturbed = perturbed.copy()
    reference["_cut"] = [cutoff_map.get((row.object_id, row.direction), -1) for row in reference.itertuples()]
    perturbed["_cut"] = [cutoff_map.get((row.object_id, row.direction), -1) for row in perturbed.itertuples()]
    reference = reference[reference["time_idx"] < reference["_cut"]]
    perturbed = perturbed[perturbed["time_idx"] < perturbed["_cut"]]
    if reference.empty or perturbed.empty:
        return {"passed": False, "errors": ["no comparable prefix rows"]}
    for name, frame in (("reference", reference), ("perturbed", perturbed)):
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                return {"passed": False, "errors": [f"{name} contains nonfinite prefix values in {column}"]}
    merged = reference[KEYS + columns].merge(
        perturbed[KEYS + columns], on=KEYS, suffixes=("_reference", "_perturbed"),
        how="outer", indicator=True, validate="one_to_one",
    )
    coverage_errors = int((merged["_merge"] != "both").sum())
    maximum = 0.0
    per_column: dict[str, float] = {}
    if coverage_errors == 0:
        for column in columns:
            difference = np.abs(
                merged[f"{column}_reference"].to_numpy(dtype=float)
                - merged[f"{column}_perturbed"].to_numpy(dtype=float)
            )
            value = float(np.nanmax(difference)) if len(difference) else 0.0
            per_column[column] = value
            maximum = max(maximum, value)
    errors = []
    if coverage_errors:
        errors.append(f"prefix key coverage mismatch rows: {coverage_errors}")
    if maximum > tolerance:
        errors.append(f"maximum prefix difference {maximum} exceeds tolerance {tolerance}")
    return {
        "passed": not errors,
        "rows_compared": int(len(merged) - coverage_errors),
        "columns_compared": columns,
        "maximum_abs_difference": maximum,
        "per_column_max_abs_difference": per_column,
        "tolerance": tolerance,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    result = audit(
        read_frame(args.reference), read_frame(args.perturbed), pd.read_csv(args.cutoffs),
        args.columns, args.tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
