"""Score submitted service actions under the SPLID-TrustBench policy contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["object_id", "time_slot"]
INTERNAL_ACTIONS = ("A", "C", "R", "D")
PUBLIC_ACTIONS = ("allow", "restrict", "restrict", "reject")
UTILITY = np.array([[1.0, 0.6, -0.4, -1.0], [-2.0, 0.2, 0.6, 1.0]])


def validate_keys(frame: pd.DataFrame, name: str) -> None:
    if not set(KEYS).issubset(frame):
        raise ValueError(f"{name} requires {KEYS}")
    for key in KEYS:
        values = pd.to_numeric(frame[key], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{name}.{key} must contain nonnegative integer indices")
        frame[key] = values.astype("int64")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{name} has duplicate object-slot keys")


def evaluate(ground_truth: pd.DataFrame, submission: pd.DataFrame) -> dict:
    """Join by object-slot key; require every target slot, including empty views.

    Actions are submitted explicitly: this evaluator neither chooses thresholds
    nor implements any producer, consumer, or historical policy.
    """
    truth = ground_truth.copy()
    prediction = submission.copy()
    for name, frame in (("ground truth", truth), ("submission", prediction)):
        validate_keys(frame, name)
    if truth.empty:
        raise ValueError("ground truth is empty")
    if "risk_label" not in truth or not truth.risk_label.isin([0, 1]).all():
        raise ValueError("risk_label must be binary")
    if "internal_action" not in prediction or not prediction.internal_action.isin(INTERNAL_ACTIONS).all():
        raise ValueError("internal_action must be one of A, C, R, D")
    columns = KEYS + ["internal_action"]
    if "public_action" in prediction:
        columns.append("public_action")
    joined = truth[KEYS + ["risk_label"]].merge(
        prediction[columns], on=KEYS, how="outer", indicator=True, validate="one_to_one"
    )
    if not joined._merge.eq("both").all():
        raise ValueError("submission keys must exactly match the target slot inventory")
    actions = pd.Categorical(joined.internal_action, categories=INTERNAL_ACTIONS).codes
    public = np.asarray(PUBLIC_ACTIONS)[actions]
    if "public_action" in joined and not np.array_equal(joined.public_action.to_numpy(), public):
        raise ValueError("public_action disagrees with the internal-to-public action mapping")
    risky = joined.risk_label.to_numpy(dtype=int)
    allowed = actions == 0
    regret = 1.0 - UTILITY[risky, actions]

    def fraction(numerator: int, denominator: int):
        return float(numerator / denominator) if denominator else None

    return {
        "slots": int(len(joined)),
        "risky_slots": int(risky.sum()),
        "normal_slots": int((risky == 0).sum()),
        "false_allowance": fraction(int(((risky == 1) & allowed).sum()), int(risky.sum())),
        "normal_restriction_rate": fraction(int(((risky == 0) & ~allowed).sum()), int((risky == 0).sum())),
        "policy_regret": float(regret.mean()),
        "public_action_counts": {name: int((public == name).sum()) for name in ("allow", "restrict", "reject")},
        "utility": {"internal_actions": list(INTERNAL_ACTIONS), "normal": UTILITY[0].tolist(), "risky": UTILITY[1].tolist()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.ground_truth), pd.read_csv(args.submission))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
