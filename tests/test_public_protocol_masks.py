"""Contract tests for the protocol-only artifact, without model dependencies."""
import numpy as np
import pandas as pd
import pytest

from benchmarks.splid_trustbench.delivery.causal_masks import CONDITIONS, delivered_view


@pytest.mark.parametrize("condition", CONDITIONS)
def test_delivered_prefix_does_not_depend_on_future_rows(condition):
    values = np.ones((64, 12, 26), np.float32)
    native = np.ones((64, 12, 2), bool)
    slots = pd.DataFrame({"object_id": 7, "time_slot": np.arange(64)})
    full = delivered_view(values, native, slots, condition, 20261017)
    prefix = delivered_view(values[:27], native[:27], slots.iloc[:27], condition, 20261017)
    for first, second in zip(full, prefix):
        np.testing.assert_array_equal(first[:27], second)


def test_native_partial_tail_is_complete_when_all_expected_tokens_arrive():
    values = np.ones((1, 12, 26), np.float32)
    native = np.zeros((1, 12, 2), bool)
    native[:, :4] = True
    slots = pd.DataFrame({"object_id": [7], "time_slot": [0]})
    _, _, incomplete, available = delivered_view(values, native, slots, "clean", 20261017)
    assert available.all() and not incomplete.any()
