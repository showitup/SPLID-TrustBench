from __future__ import annotations

import pandas as pd

from benchmarks.splid_trustbench.availability.evaluate_submission import evaluate


TRACKS = {
    "tracks": {
        "bounded_6": {
            "maximum_lookahead_steps": 0,
            "maximum_detection_delay_steps": 6,
            "requires_future_suffix_audit": True,
        }
    }
}


def metadata() -> dict:
    return {
        "availability_track": "bounded_6",
        "lookahead_budget_steps": 0,
        "teacher_used_at_inference": False,
        "causal_audit": {
            "performed": True,
            "passed": True,
            "layered": True,
            "future_perturbation_max_abs_difference": 0.0,
            "raw_encoder_future_suffix": {"performed": True, "passed": True},
            "commit_layer_replay": {"performed": True, "passed": True},
        },
        "evaluation_status": "observed_development_test",
        "sealed_claim": False,
    }


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.DataFrame([
        {"ObjectID": 1, "TimeIndex": 0, "Direction": "EW", "Node": "SS", "Type": "NK"},
        {"ObjectID": 1, "TimeIndex": 10, "Direction": "EW", "Node": "ID", "Type": "CK"},
    ])
    submission = truth.copy()
    submission["available_at"] = [6, 13]
    return truth, submission


def test_bounded_submission_passes_and_scores() -> None:
    truth, submission = frames()
    result = evaluate(truth, submission, metadata(), TRACKS, [1])
    assert result["all_checks_passed"]
    assert result["official_metric"]["f2"] == 1.0


def test_excess_delay_fails_closed() -> None:
    truth, submission = frames()
    submission.loc[1, "available_at"] = 17
    result = evaluate(truth, submission, metadata(), TRACKS, [1])
    assert not result["all_checks_passed"]
    assert "maximum detection delay" in result["errors"][0]


def test_missing_causal_audit_fails_closed() -> None:
    truth, submission = frames()
    bad_metadata = metadata()
    bad_metadata["causal_audit"]["passed"] = False
    result = evaluate(truth, submission, bad_metadata, TRACKS, [1])
    assert not result["all_checks_passed"]


def test_missing_commit_layer_audit_fails_closed() -> None:
    truth, submission = frames()
    bad_metadata = metadata()
    bad_metadata["causal_audit"]["commit_layer_replay"]["passed"] = False
    result = evaluate(truth, submission, bad_metadata, TRACKS, [1])
    assert not result["all_checks_passed"]
    assert any("commit-layer" in error for error in result["errors"])


def test_commit_after_finite_horizon_fails_closed() -> None:
    truth, submission = frames()
    result = evaluate(truth, submission, metadata(), TRACKS, [1], {1: 12})
    assert not result["all_checks_passed"]
    assert any("finite observation horizon" in error for error in result["errors"])


def support_contract():
    tracks = {"tracks": {"bounded_support_6": {
        "maximum_lookahead_steps": 0, "maximum_detection_delay_steps": None,
        "maximum_resolution_delay_steps": 6,
        "resolution_origin_column": "first_support_at",
        "requires_future_suffix_audit": True,
    }}}
    meta = metadata()
    meta["availability_track"] = "bounded_support_6"
    truth, submission = frames()
    submission["first_support_at"] = [0, 13]
    submission["available_at"] = [6, 19]
    return truth, submission, meta, tracks


def test_support_deadline_is_distinct_from_event_delay():
    truth, submission, meta, tracks = support_contract()
    result = evaluate(truth, submission, meta, tracks, [1], {1: 30})
    assert result["all_checks_passed"]
    assert result["emitted_event_latency"]["event_relative_max_steps"] == 9
    assert result["emitted_event_latency"]["first_support_max_steps"] == 6
    submission.loc[1, "available_at"] = 20
    result = evaluate(truth, submission, meta, tracks, [1], {1: 30})
    assert not result["all_checks_passed"]
    assert any("first-support resolution delay" in error for error in result["errors"])


def test_support_contract_requires_real_support_timestamp():
    truth, submission, meta, tracks = support_contract()
    result = evaluate(truth, submission.drop(columns="first_support_at"), meta, tracks, [1])
    assert not result["all_checks_passed"]
    submission.loc[1, "first_support_at"] = 20
    assert not evaluate(truth, submission, meta, tracks, [1])["all_checks_passed"]


def test_fractional_release_is_not_silently_truncated():
    truth, submission = frames()
    submission["available_at"] = [6., 13.5]
    result = evaluate(truth, submission, metadata(), TRACKS, [1])
    assert not result["all_checks_passed"]
    assert any("integer indices" in error for error in result["errors"])
