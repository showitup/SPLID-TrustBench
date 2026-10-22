#!/usr/bin/env python3
"""Validate orbit_evidence.parquet and orbit_evidence.meta.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


REQUIRED_COLUMNS = [
    "object_id",
    "time_idx",
    "direction",
    "timestamp_utc",
    "split",
    "p_background",
    "p_start",
    "p_inside",
    "p_end",
    "p_id",
    "p_ad",
    "p_ik",
    "p_mode_nk",
    "p_mode_ck",
    "p_mode_ek",
    "p_mode_hk",
    "behavior_consistency",
    "forecast_residual",
    "physics_residual",
    "uncertainty",
    "event_logit_margin",
    "mode_entropy",
    "is_valid",
]
V11_REQUIRED_COLUMNS = [
    "drift_residual",
    "physics_violation",
    "drift_risk",
    "physics_risk",
    "evidence_schema_version",
]
KEY_COLUMNS = ["object_id", "time_idx", "direction"]
VALID_DIRECTIONS = {"EW", "NS"}
VALID_SPLITS = {"train", "val", "test", "stress_train", "stress_val", "stress_test"}
BOUNDARY_PROBS = ["p_background", "p_start", "p_inside", "p_end"]
EVENT_PROBS = ["p_id", "p_ad", "p_ik"]
MODE_PROBS = ["p_mode_nk", "p_mode_ck", "p_mode_ek", "p_mode_hk"]
PROBABILITY_COLUMNS = BOUNDARY_PROBS + EVENT_PROBS + MODE_PROBS + ["behavior_consistency", "uncertainty"]
V11_PROBABILITY_COLUMNS = ["drift_risk", "physics_risk"]
NON_NEGATIVE_COLUMNS = ["forecast_residual", "physics_residual", "mode_entropy"]
V11_NON_NEGATIVE_COLUMNS = ["drift_residual", "physics_violation"]
COMMITMENT_COLUMNS = [
    "committed_event",
    "committed_event_score",
    "committed_available_at_time_idx",
    "committed_event_time_idx",
    "committed_node",
    "committed_following_mode",
    "committed_latency_steps",
]
VALID_COMMITTED_NODES = {"ID", "AD", "IK"}
VALID_COMMITTED_MODES = {"NK", "CK", "EK", "HK"}
SCORE_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in KEY_COLUMNS + ["timestamp_utc", "split", "is_valid"]]
REQUIRED_META_FIELDS = [
    "schema_version",
    "producer_name",
    "producer_version",
    "source_split_manifest",
    "label_protocol_version",
    "feature_spec_version",
    "decoder_config",
    "calibration_config",
    "normalization_stats_source",
    "created_at",
    "evidence_file",
    "row_count",
    "object_count",
    "invalid_row_policy",
    "invalid_rows",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate orbit_evidence parquet and metadata.")
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--object-manifest", type=Path)
    p.add_argument("--allow-subset", action="store_true", help="Allow evidence to cover a subset of benchmark-valid objects.")
    p.add_argument("--prob-tol", type=float, default=1e-4)
    p.add_argument("--output-report", type=Path)
    return p.parse_args()


def add_error(errors: List[Dict], code: str, detail: str) -> None:
    errors.append({"code": code, "detail": detail})


def add_warning(warnings: List[Dict], code: str, detail: str) -> None:
    warnings.append({"code": code, "detail": detail})


def finite_non_null_mask(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    return ~df[columns].isna().any(axis=1)


def validate_meta(meta: Dict, df: pd.DataFrame, errors: List[Dict], warnings: List[Dict]) -> None:
    missing = [field for field in REQUIRED_META_FIELDS if field not in meta]
    if missing:
        add_error(errors, "meta_missing_required_fields", f"Missing fields: {missing}")

    if meta.get("row_count") != len(df):
        add_error(errors, "meta_row_count_mismatch", f"meta row_count={meta.get('row_count')} actual={len(df)}")

    if meta.get("object_count") != int(df["object_id"].nunique()):
        add_error(
            errors,
            "meta_object_count_mismatch",
            f"meta object_count={meta.get('object_count')} actual={int(df['object_id'].nunique())}",
        )

    invalid_rows = int((df["is_valid"] == 0).sum()) if "is_valid" in df else None
    if meta.get("invalid_rows") != invalid_rows:
        add_error(errors, "meta_invalid_rows_mismatch", f"meta invalid_rows={meta.get('invalid_rows')} actual={invalid_rows}")

    policy = meta.get("invalid_row_policy", {})
    if not isinstance(policy, dict):
        add_error(errors, "meta_invalid_row_policy_not_object", "invalid_row_policy must be a JSON object.")
    else:
        for field in ["row_retention", "null_score_fields_allowed_if_is_valid_zero", "reason_tracking"]:
            if field not in policy:
                add_error(errors, "meta_invalid_policy_missing_field", f"Missing invalid_row_policy.{field}")

    if meta.get("schema_version") not in {"1.0.0", "1.1.0"}:
        add_warning(
            warnings,
            "meta_schema_version_unexpected",
            f"Expected schema_version in {{1.0.0, 1.1.0}}, got {meta.get('schema_version')}",
        )


def evidence_schema_version(meta: Dict) -> str:
    return str(meta.get("schema_version", "1.0.0"))


def required_columns_for(meta: Dict) -> List[str]:
    columns = list(REQUIRED_COLUMNS)
    if evidence_schema_version(meta) >= "1.1.0":
        columns.extend(V11_REQUIRED_COLUMNS)
    return columns


def validate_columns(df: pd.DataFrame, meta: Dict, errors: List[Dict]) -> None:
    missing = [c for c in required_columns_for(meta) if c not in df.columns]
    if missing:
        add_error(errors, "missing_required_columns", f"Missing columns: {missing}")
        return

    null_keys = df[KEY_COLUMNS].isna().any(axis=1).sum()
    if int(null_keys) > 0:
        add_error(errors, "null_key_values", f"Rows with null key values: {int(null_keys)}")

    duplicated = df.duplicated(subset=KEY_COLUMNS).sum()
    if int(duplicated) > 0:
        add_error(errors, "duplicate_primary_key", f"Duplicate primary key rows: {int(duplicated)}")


def validate_values(df: pd.DataFrame, meta: Dict, prob_tol: float, errors: List[Dict]) -> None:
    bad_direction = ~df["direction"].isin(VALID_DIRECTIONS)
    if bool(bad_direction.any()):
        add_error(errors, "invalid_direction", f"Invalid direction rows: {int(bad_direction.sum())}")

    bad_split = ~df["split"].isin(VALID_SPLITS)
    if bool(bad_split.any()):
        add_error(errors, "invalid_split", f"Invalid split rows: {int(bad_split.sum())}")

    bad_is_valid = ~df["is_valid"].isin([0, 1])
    if bool(bad_is_valid.any()):
        add_error(errors, "invalid_is_valid", f"Invalid is_valid rows: {int(bad_is_valid.sum())}")

    valid_rows = df["is_valid"] == 1
    valid_nulls = df.loc[valid_rows, SCORE_COLUMNS].isna().any(axis=1)
    if bool(valid_nulls.any()):
        add_error(errors, "valid_rows_have_null_scores", f"Valid rows with null score fields: {int(valid_nulls.sum())}")

    null_score_rows = df[SCORE_COLUMNS].isna().any(axis=1)
    null_but_valid = null_score_rows & valid_rows
    if bool(null_but_valid.any()):
        add_error(errors, "null_scores_marked_valid", f"Rows with null scores and is_valid=1: {int(null_but_valid.sum())}")

    probability_columns = list(PROBABILITY_COLUMNS)
    non_negative_columns = list(NON_NEGATIVE_COLUMNS)
    if evidence_schema_version(meta) >= "1.1.0":
        probability_columns.extend(V11_PROBABILITY_COLUMNS)
        non_negative_columns.extend(V11_NON_NEGATIVE_COLUMNS)
    if "evidence_reliability" in df.columns:
        probability_columns.append("evidence_reliability")

    for col in probability_columns:
        s = df[col].dropna()
        bad = (s < 0) | (s > 1)
        if bool(bad.any()):
            add_error(errors, "probability_out_of_range", f"{col} out-of-range values: {int(bad.sum())}")

    for col in non_negative_columns:
        s = df[col].dropna()
        bad = s < 0
        if bool(bad.any()):
            add_error(errors, "negative_value", f"{col} negative values: {int(bad.sum())}")

    complete_valid = valid_rows & finite_non_null_mask(df, BOUNDARY_PROBS + EVENT_PROBS + MODE_PROBS)
    groups = {
        "boundary_probability_sum": BOUNDARY_PROBS,
        "event_probability_sum": EVENT_PROBS,
        "mode_probability_sum": MODE_PROBS,
    }
    for code, columns in groups.items():
        sums = df.loc[complete_valid, columns].sum(axis=1)
        bad = (sums - 1.0).abs() > prob_tol
        if bool(bad.any()):
            add_error(errors, code, f"Rows outside tolerance {prob_tol}: {int(bad.sum())}")


def validate_commitments(df: pd.DataFrame, meta: Dict, errors: List[Dict]) -> None:
    present = [column for column in COMMITMENT_COLUMNS if column in df.columns]
    if not present:
        return
    missing = [column for column in COMMITMENT_COLUMNS if column not in df.columns]
    if missing:
        add_error(
            errors,
            "incomplete_commitment_contract",
            f"Commitment fields are atomic; missing columns: {missing}",
        )
        return

    bad_flag = ~df["committed_event"].isin([0, 1])
    if bool(bad_flag.any()):
        add_error(errors, "invalid_committed_event", f"Invalid rows: {int(bad_flag.sum())}")
    committed = df["committed_event"].eq(1)
    if not bool(committed.any()):
        return

    values = df.loc[committed]
    nulls = values[COMMITMENT_COLUMNS].isna().any(axis=1)
    if bool(nulls.any()):
        add_error(errors, "null_committed_fields", f"Committed rows with null fields: {int(nulls.sum())}")
        return
    bad_score = ~values["committed_event_score"].between(0.0, 1.0)
    if bool(bad_score.any()):
        add_error(errors, "committed_score_out_of_range", f"Invalid rows: {int(bad_score.sum())}")
    bad_node = ~values["committed_node"].isin(VALID_COMMITTED_NODES)
    if bool(bad_node.any()):
        add_error(errors, "invalid_committed_node", f"Invalid rows: {int(bad_node.sum())}")
    bad_mode = ~values["committed_following_mode"].isin(VALID_COMMITTED_MODES)
    if bool(bad_mode.any()):
        add_error(errors, "invalid_committed_mode", f"Invalid rows: {int(bad_mode.sum())}")

    for column in ("time_idx", "committed_event_time_idx", "committed_available_at_time_idx", "committed_latency_steps"):
        numeric = pd.to_numeric(values[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any() or (numeric % 1 != 0).any():
            add_error(errors, "noninteger_commitment_timing", f"{column} must contain nonnegative integer indices.")
            return
    event_time = values["committed_event_time_idx"].astype(int)
    available_at = values["committed_available_at_time_idx"].astype(int)
    latency = values["committed_latency_steps"].astype(int)
    mismatch = (
        event_time.ne(values["time_idx"].astype(int))
        | latency.ne(available_at - event_time)
        | latency.lt(0)
    )
    if bool(mismatch.any()):
        add_error(errors, "invalid_commitment_timing", f"Invalid rows: {int(mismatch.sum())}")

    contract = meta.get("availability_contract", {})
    if isinstance(contract, dict) and contract.get("resolution_origin") == "first_support":
        column = "committed_first_support_time_idx"
        maximum = contract.get("maximum_resolution_delay_steps")
        if column not in values or maximum is None:
            add_error(errors, "missing_resolution_contract", "First-support resolution requires its timestamp and budget.")
            return
        support = pd.to_numeric(values[column], errors="coerce")
        if support.isna().any() or (support < 0).any() or (support % 1 != 0).any():
            add_error(errors, "invalid_first_support", "First-support timestamps must be nonnegative integer indices.")
        elif (support > available_at).any() or ((available_at - support) > int(maximum)).any():
            add_error(errors, "resolution_budget_exceeded", f"First-support resolution exceeds {maximum} steps or has reversed timing.")
        return
    maximum = contract.get("online_maximum_lookahead") if isinstance(contract, dict) else None
    if maximum is not None:
        over_budget = latency.gt(int(maximum))
        if bool(over_budget.any()):
            add_error(
                errors,
                "commitment_latency_budget_exceeded",
                f"Maximum={maximum}, invalid rows: {int(over_budget.sum())}",
            )


def validate_coverage(
    df: pd.DataFrame,
    object_manifest_path: Optional[Path],
    allow_subset: bool,
    errors: List[Dict],
    warnings: List[Dict],
) -> None:
    if object_manifest_path is None:
        add_warning(warnings, "coverage_not_checked", "No object manifest supplied.")
        return

    manifest = pd.read_csv(object_manifest_path)
    valid_manifest = manifest[manifest["benchmark_valid"] == 1].copy()
    seq_len = valid_manifest.set_index("object_id")["seq_len"].to_dict()
    evidence_object_ids = set(df["object_id"].unique().tolist())
    valid_object_ids = set(valid_manifest["object_id"].unique().tolist())
    malformed_ids = set(manifest[manifest["benchmark_valid"] == 0]["object_id"].unique().tolist())

    malformed_present = evidence_object_ids & malformed_ids
    if malformed_present:
        add_error(errors, "malformed_objects_present", f"Malformed object ids in evidence: {sorted(malformed_present)[:20]}")

    unknown_ids = evidence_object_ids - valid_object_ids
    if unknown_ids:
        add_error(errors, "unknown_object_ids", f"Unknown object ids in evidence: {sorted(unknown_ids)[:20]}")

    if not allow_subset:
        missing_ids = valid_object_ids - evidence_object_ids
        if missing_ids:
            add_error(errors, "missing_benchmark_valid_objects", f"Missing valid object ids: count={len(missing_ids)}")
    else:
        add_warning(warnings, "subset_coverage_mode", "Coverage is checked only for objects present in the evidence file.")

    expected_ids = evidence_object_ids if allow_subset else valid_object_ids
    counts = df.groupby(["object_id", "direction"]).size().reset_index(name="n")
    for object_id in sorted(expected_ids):
        if object_id not in seq_len:
            continue
        for direction in ["EW", "NS"]:
            rows = counts[(counts["object_id"] == object_id) & (counts["direction"] == direction)]
            actual = int(rows["n"].iloc[0]) if not rows.empty else 0
            expected = int(seq_len[object_id])
            if actual != expected:
                add_error(
                    errors,
                    "stream_row_count_mismatch",
                    f"object_id={object_id} direction={direction} expected={expected} actual={actual}",
                )

    for object_id, group in df.groupby("object_id"):
        expected = int(seq_len.get(int(object_id), 0))
        if expected <= 0:
            continue
        bad_time = (group["time_idx"] < 0) | (group["time_idx"] >= expected)
        if bool(bad_time.any()):
            add_error(errors, "time_idx_out_of_range", f"object_id={object_id} rows={int(bad_time.sum())}")


def main() -> None:
    args = parse_args()
    errors: List[Dict] = []
    warnings: List[Dict] = []

    df = pd.read_parquet(args.evidence)
    meta = json.loads(args.meta.read_text(encoding="utf-8"))

    validate_columns(df, meta, errors)
    if not any(err["code"] == "missing_required_columns" for err in errors):
        validate_meta(meta, df, errors, warnings)
        validate_values(df, meta, args.prob_tol, errors)
        validate_commitments(df, meta, errors)
        validate_coverage(df, args.object_manifest, args.allow_subset, errors, warnings)

    report = {
        "evidence": str(args.evidence),
        "meta": str(args.meta),
        "row_count": int(len(df)),
        "object_count": int(df["object_id"].nunique()) if "object_id" in df else None,
        "valid_rows": int((df["is_valid"] == 1).sum()) if "is_valid" in df else None,
        "invalid_rows": int((df["is_valid"] == 0).sum()) if "is_valid" in df else None,
        "allow_subset": bool(args.allow_subset),
        "errors": errors,
        "warnings": warnings,
        "all_checks_passed": len(errors) == 0,
    }

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=True, indent=2))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
