from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.splid_trustbench.delivery.transport import (
    DegradationCondition,
    apply_condition,
    collapse_for_control,
    directional_tokens,
    typed_clean,
)


def inputs() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x = np.arange(5 * 2 * 26, dtype=np.float32).reshape(5, 2, 26)
    presence = np.ones((5, 2, 2), dtype=bool)
    slots = pd.DataFrame({"object_id": [1] * 5, "time_slot": range(5), "split": ["val"] * 5})
    values, token_presence = directional_tokens(x, presence)
    return values, token_presence, slots


def test_typed_channels_round_trip_to_control_tokens() -> None:
    values, presence, _ = inputs()
    clean = typed_clean(values, presence)
    restored, restored_presence = collapse_for_control(clean)
    np.testing.assert_allclose(restored, values)
    np.testing.assert_array_equal(restored_presence, presence)


def test_clean_delivery_is_independent_of_native_presence() -> None:
    values, presence, _ = inputs()
    presence[0, 0] = False
    clean = typed_clean(values, presence)
    assert not clean.presence[0, 0].any()
    assert clean.availability[0, 0].all()


def test_delay_uses_past_only_and_records_age() -> None:
    values, presence, slots = inputs()
    delayed = apply_condition(values, presence, slots, DegradationCondition("delay_01", "delay", "1", 1), 7)
    assert not delayed.presence[0].any()
    np.testing.assert_allclose(delayed.values[1], typed_clean(values, presence).values[0])
    assert np.all(delayed.age[1] == 1)


def test_outage_is_label_independent_and_carries_stale_evidence() -> None:
    values, presence, slots = inputs()
    condition = DegradationCondition("slot_outage_03", "slot_outage", "3", 3)
    first = apply_condition(values, presence, slots, condition, 11)
    changed = slots.copy()
    changed["slot_risk_label"] = [1, 0, 1, 0, 1]
    second = apply_condition(values, presence, changed, condition, 11)
    np.testing.assert_array_equal(first.outage_mask, second.outage_mask)
    assert first.outage_mask.sum() == 3
    assert not first.availability[first.outage_mask].any()


def test_random_missingness_is_deterministic() -> None:
    values, presence, slots = inputs()
    condition = DegradationCondition("token_missing_30", "token_missing", "30%", 0.3)
    first = apply_condition(values, presence, slots, condition, 123)
    second = apply_condition(values, presence, slots, condition, 123)
    np.testing.assert_array_equal(first.presence, second.presence)
