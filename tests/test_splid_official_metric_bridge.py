"""Benchmark-only definitions extracted from tests/test_splid_official_metric_bridge.py."""


import pandas as pd
import pytest
from benchmarks.splid_trustbench.official_metric_bridge import score_official_nodes


def frame(rows):
    return pd.DataFrame(rows, columns=["ObjectID", "TimeIndex", "Direction", "Node", "Type"])


def test_official_matching_preserves_wrong_class_semantics() -> None:
    ground_truth = frame([(1, 0, "EW", "SS", "NK"), (1, 10, "EW", "ID", "CK")])
    participant = frame([(1, 0, "EW", "SS", "NK"), (1, 12, "EW", "IK", "CK")])
    score, per_object, _ = score_official_nodes(ground_truth, participant, tolerance=6)
    assert (score.tp, score.fp, score.fn) == (1, 1, 0)
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(1.0)
    assert per_object.iloc[0]["f2"] == pytest.approx(5 / 6)


def test_official_matching_counts_unmatched_prediction() -> None:
    ground_truth = frame([(1, 0, "NS", "SS", "HK")])
    participant = frame([(1, 0, "NS", "SS", "HK"), (1, 30, "NS", "AD", "NK")])
    score, _, _ = score_official_nodes(ground_truth, participant)
    assert (score.tp, score.fp, score.fn) == (1, 1, 0)
