import numpy as np
import pandas as pd
import pytest

from benchmarks.splid_trustbench.availability.audit_future_suffix import audit


def inputs():
    reference = pd.DataFrame({"object_id": [1, 1], "direction": ["EW", "EW"],
                              "time_idx": [0, 1], "score": [.2, .4]})
    cutoffs = pd.DataFrame({"object_id": [1], "direction": ["EW"], "cut_time_idx": [1]})
    return reference, cutoffs


def test_suffix_changes_do_not_affect_prefix_audit():
    reference, cutoffs = inputs()
    changed = reference.copy()
    changed.loc[1, "score"] = .9
    report = audit(reference, changed, cutoffs, ["score"], 1e-6)
    assert report["passed"] and report["rows_compared"] == 1


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, .3])
def test_invalid_or_changed_prefix_does_not_pass(value):
    reference, cutoffs = inputs()
    changed = reference.copy()
    changed.loc[0, "score"] = value
    assert not audit(reference, changed, cutoffs, ["score"], 1e-6)["passed"]


def test_empty_prefix_does_not_pass():
    reference, cutoffs = inputs()
    cutoffs["cut_time_idx"] = 0
    assert not audit(reference, reference, cutoffs, ["score"], 1e-6)["passed"]


def test_duplicate_cutoff_does_not_pass():
    reference, cutoffs = inputs()
    assert not audit(reference, reference, pd.concat([cutoffs, cutoffs]), ["score"], 1e-6)["passed"]
