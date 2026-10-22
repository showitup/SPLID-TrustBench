#!/usr/bin/env python3
"""Fail-closed evaluator for availability-aware SPLID sparse-node submissions."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.splid_trustbench.official_metric_bridge import (  # noqa: E402
    OFFICIAL_COLUMNS,
    score_official_nodes,
)


AVAILABILITY_COLUMNS = [*OFFICIAL_COLUMNS, "available_at"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path(__file__).with_name("tracks.json"),
    )
    parser.add_argument("--object-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_object_manifest(path: Path | None) -> tuple[list[int] | None, dict[int, int] | None]:
    if path is None:
        return None, None
    frame = pd.read_csv(path)
    id_column = "object_id" if "object_id" in frame.columns else "ObjectID"
    if "benchmark_split" in frame.columns:
        frame = frame[frame["benchmark_split"] == "test"]
    if "benchmark_valid" in frame.columns:
        frame = frame[frame["benchmark_valid"].astype(int) == 1]
    object_ids = frame[id_column].astype(int).tolist()
    horizons = None
    if "seq_len" in frame.columns:
        horizons = {
            int(object_id): int(seq_len) - 1
            for object_id, seq_len in zip(frame[id_column], frame["seq_len"])
        }
    return object_ids, horizons


def validate_availability(
    submission: pd.DataFrame,
    metadata: dict,
    track: dict,
    object_horizons: dict[int, int] | None = None,
) -> list[str]:
    errors: list[str] = []
    missing = [column for column in AVAILABILITY_COLUMNS if column not in submission.columns]
    if missing:
        return [f"missing submission columns: {missing}"]
    if submission[AVAILABILITY_COLUMNS].isna().any().any():
        errors.append("submission contains null required fields")
    for column in ("ObjectID", "TimeIndex", "available_at"):
        values = pd.to_numeric(submission[column], errors="coerce")
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            errors.append(f"{column} must contain nonnegative integer indices")
    if errors:
        return errors
    if submission.duplicated(OFFICIAL_COLUMNS).any():
        errors.append("submission contains duplicate official rows")
    if not submission["Direction"].isin(["EW", "NS", "ES"]).all():
        errors.append("submission contains invalid Direction")
    if (submission["available_at"].astype(int) < submission["TimeIndex"].astype(int)).any():
        errors.append("available_at precedes reported event TimeIndex")
    if object_horizons is not None:
        unknown = sorted(
            set(submission["ObjectID"].astype(int)).difference(object_horizons)
        )
        if unknown:
            errors.append(f"submission contains objects without a finite observation horizon: {unknown[:5]}")
        else:
            horizons = submission["ObjectID"].astype(int).map(object_horizons).astype(int)
            if (submission["available_at"].astype(int) > horizons).any():
                errors.append("submission contains commits after the finite observation horizon")

    maximum_delay = track["maximum_detection_delay_steps"]
    if maximum_delay is not None:
        delays = submission["available_at"].astype(int) - submission["TimeIndex"].astype(int)
        if (delays > int(maximum_delay)).any():
            errors.append(f"submission exceeds maximum detection delay {maximum_delay}")
    resolution_budget = track.get("maximum_resolution_delay_steps")
    if resolution_budget is not None:
        origin = track["resolution_origin_column"]
        if origin not in submission:
            errors.append(f"resolution contract requires {origin}")
        else:
            support = pd.to_numeric(submission[origin], errors="coerce")
            release = submission["available_at"].astype(int)
            if support.isna().any() or (support < 0).any() or (support % 1 != 0).any():
                errors.append(f"{origin} must contain nonnegative integer indices")
            elif (support > release).any():
                errors.append(f"{origin} occurs after resolution")
            elif ((release - support) > resolution_budget).any():
                errors.append(f"submission exceeds first-support resolution delay {resolution_budget}")
    expected_lookahead = track["maximum_lookahead_steps"]
    if metadata.get("lookahead_budget_steps") != expected_lookahead:
        errors.append(
            f"lookahead budget mismatch: {metadata.get('lookahead_budget_steps')} != {expected_lookahead}"
        )
    if metadata.get("sealed_claim") is not False:
        errors.append("sealed_claim must be false for the observed development cohort")
    if metadata.get("evaluation_status") != "observed_development_test":
        errors.append("evaluation_status must be observed_development_test")
    if track["requires_future_suffix_audit"]:
        audit = metadata.get("causal_audit", {})
        if not audit.get("performed") or not audit.get("passed"):
            errors.append("causal track requires a passing future-suffix audit")
        if audit.get("layered") is not True:
            errors.append("causal track requires a layered encoder-and-commit audit")
        encoder = audit.get("raw_encoder_future_suffix", {})
        commit = audit.get("commit_layer_replay", {})
        if not encoder.get("performed") or not encoder.get("passed"):
            errors.append("causal track requires a passing raw-encoder future-suffix audit")
        if not commit.get("performed") or not commit.get("passed"):
            errors.append("causal track requires a passing commit-layer replay audit")
        if metadata.get("teacher_used_at_inference"):
            errors.append("causal track forbids an offline teacher at inference")
    return errors


def evaluate(
    ground_truth: pd.DataFrame,
    submission: pd.DataFrame,
    metadata: dict,
    track_config: dict,
    object_ids: list[int] | None,
    object_horizons: dict[int, int] | None = None,
) -> dict:
    track_name = metadata.get("availability_track")
    if track_name not in track_config["tracks"]:
        return {"all_checks_passed": False, "errors": [f"unknown availability track: {track_name}"]}
    errors = validate_availability(
        submission, metadata, track_config["tracks"][track_name], object_horizons
    )
    if errors:
        return {"all_checks_passed": False, "availability_track": track_name, "errors": errors}
    score, per_object, matches = score_official_nodes(
        ground_truth, submission[OFFICIAL_COLUMNS], tolerance=6, object_ids=object_ids
    )
    counts = matches["classification"].value_counts()
    event_rows = submission[submission["Node"].isin(["ID", "AD", "IK"])]
    event_delays = event_rows["available_at"].astype(int) - event_rows["TimeIndex"].astype(int)
    delay_report = {"emitted_event_count": int(len(event_rows))}
    if len(event_rows):
        delay_report.update(event_relative_mean_steps=float(event_delays.mean()),
                            event_relative_max_steps=int(event_delays.max()))
        origin = track_config["tracks"][track_name].get("resolution_origin_column")
        if origin:
            delays = event_rows["available_at"].astype(int) - event_rows[origin].astype(int)
            delay_report.update(first_support_mean_steps=float(delays.mean()),
                                first_support_max_steps=int(delays.max()))
    return {
        "all_checks_passed": True,
        "availability_track": track_name,
        "official_metric": asdict(score),
        "emitted_event_latency": delay_report,
        "error_counts": {
            "tp": int(counts.get("TP", 0)),
            "fp_class": int(counts.get("FP_CLASS", 0)),
            "fp_unmatched": int(counts.get("FP_UNMATCHED", 0)),
            "fn": int(counts.get("FN", 0)),
        },
        "per_object_count": len(per_object),
        "errors": [],
    }


def main() -> None:
    args = parse_args()
    ground_truth = pd.read_csv(args.ground_truth)
    submission = pd.read_csv(args.submission)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    tracks = json.loads(args.tracks.read_text(encoding="utf-8"))
    object_ids, horizons = load_object_manifest(args.object_manifest)
    result = evaluate(ground_truth, submission, metadata, tracks, object_ids, horizons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
