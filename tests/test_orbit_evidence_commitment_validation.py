from __future__ import annotations

import pandas as pd

from interfaces.orbit_evidence.validate_orbit_evidence import validate_commitments


def commitment_frame(latency: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "time_idx": 10,
                "committed_event": 1,
                "committed_event_score": 0.9,
                "committed_available_at_time_idx": 10 + latency,
                "committed_event_time_idx": 10,
                "committed_node": "IK",
                "committed_following_mode": "HK",
                "committed_latency_steps": latency,
            }
        ]
    )


def test_valid_bounded_commitment_passes() -> None:
    errors: list[dict] = []
    validate_commitments(
        commitment_frame(),
        {"availability_contract": {"online_maximum_lookahead": 6}},
        errors,
    )
    assert errors == []


def test_over_budget_commitment_fails() -> None:
    errors: list[dict] = []
    validate_commitments(
        commitment_frame(7),
        {"availability_contract": {"online_maximum_lookahead": 6}},
        errors,
    )
    assert any(error["code"] == "commitment_latency_budget_exceeded" for error in errors)


def test_partial_commitment_contract_fails() -> None:
    errors: list[dict] = []
    validate_commitments(
        pd.DataFrame([{"time_idx": 10, "committed_event": 1}]),
        {},
        errors,
    )
    assert any(error["code"] == "incomplete_commitment_contract" for error in errors)


def test_first_support_contract_keeps_event_latency_separate():
    frame = commitment_frame(9)
    frame["committed_first_support_time_idx"] = 13
    meta = {"availability_contract": {"resolution_origin": "first_support",
            "maximum_resolution_delay_steps": 6, "online_maximum_lookahead": 0}}
    errors = []
    validate_commitments(frame, meta, errors)
    assert errors == []
    frame["committed_first_support_time_idx"] = 12
    validate_commitments(frame, meta, errors)
    assert any(error["code"] == "resolution_budget_exceeded" for error in errors)


def test_fractional_commitment_time_is_rejected():
    frame = commitment_frame()
    frame["committed_available_at_time_idx"] = [16.5]
    errors = []
    validate_commitments(frame, {}, errors)
    assert any(error["code"] == "noninteger_commitment_timing" for error in errors)
