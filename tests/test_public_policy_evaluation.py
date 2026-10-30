"""Policy-contract checks independent of model implementations."""
import numpy as np
import pandas as pd
import pytest

from benchmarks.splid_trustbench.evaluate_policy import evaluate


def fixture():
    truth = pd.DataFrame({"object_id": [7] * 4, "time_slot": [0, 1, 2, 3], "risk_label": [0, 0, 1, 1]})
    submission = truth[["object_id", "time_slot"]].assign(internal_action=["A", "C", "A", "D"])
    return truth, submission


def test_contract_and_key_order():
    truth, submission = fixture()
    result = evaluate(truth, submission.iloc[::-1])
    assert result["false_allowance"] == .5
    assert result["normal_restriction_rate"] == .5
    assert result["policy_regret"] == pytest.approx(.85)
    assert result["public_action_counts"] == {"allow": 2, "restrict": 1, "reject": 1}


def test_both_internal_restrictions_share_public_action_but_not_utility():
    truth, submission = fixture()
    challenge = evaluate(truth, submission.assign(internal_action="C", public_action="restrict"))
    restrict = evaluate(truth, submission.assign(internal_action="R", public_action="restrict"))
    assert challenge["public_action_counts"] == restrict["public_action_counts"]
    assert challenge["policy_regret"] == pytest.approx(.6)
    assert restrict["policy_regret"] == pytest.approx(.9)


@pytest.mark.parametrize("issue", ["missing", "extra", "duplicate", "fractional", "infinite", "bad_action", "public_mapping"])
def test_reject_invalid_submission(issue):
    truth, submission = fixture()
    if issue == "missing":
        submission = submission.iloc[:-1]
    elif issue == "extra":
        submission = pd.concat([submission, submission.iloc[:1].assign(time_slot=9)])
    elif issue == "duplicate":
        submission = pd.concat([submission, submission.iloc[:1]])
    elif issue == "fractional":
        submission = submission.astype({"time_slot": float})
        submission.loc[0, "time_slot"] = .5
    elif issue == "infinite":
        submission = submission.astype({"object_id": float})
        submission.loc[0, "object_id"] = np.inf
    elif issue == "bad_action":
        submission.loc[0, "internal_action"] = "reject"
    else:
        submission["public_action"] = "allow"
    with pytest.raises(ValueError):
        evaluate(truth, submission)


def test_absent_class_returns_null_not_zero():
    truth, submission = fixture()
    assert evaluate(truth.assign(risk_label=0), submission)["false_allowance"] is None
    assert evaluate(truth.assign(risk_label=1), submission)["normal_restriction_rate"] is None
