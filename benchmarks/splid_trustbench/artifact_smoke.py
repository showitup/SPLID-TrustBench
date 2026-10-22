"""Read-only installation check for the public benchmark protocol artifact."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from benchmarks.splid_trustbench.availability.evaluate_submission import evaluate
from benchmarks.splid_trustbench.delivery.causal_masks import delivered_view


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = pd.read_csv(root / "benchmarks/splid_trustbench/manifests/phase2_objects.csv")
    valid = manifest.loc[manifest.benchmark_valid.eq(1)]
    counts = valid.groupby("benchmark_split").size().to_dict()
    assert counts == {"test": 498, "train": 1520, "val": 380}, counts
    truth = pd.DataFrame([(1, 0, "EW", "SS", "NK"), (1, 10, "EW", "ID", "CK")],
                         columns=["ObjectID", "TimeIndex", "Direction", "Node", "Type"])
    submission = truth.assign(available_at=20)
    meta = {"availability_track": "offline_full", "lookahead_budget_steps": None,
            "sealed_claim": False, "evaluation_status": "observed_development_test"}
    tracks = json.loads((root / "benchmarks/splid_trustbench/availability/tracks.json").read_text())
    result = evaluate(truth, submission, meta, tracks, [1], {1: 20})
    assert result["all_checks_passed"] and result["official_metric"]["f2"] == 1.
    values = np.ones((64, 12, 26), np.float32)
    present = np.ones((64, 12, 2), bool)
    slots = pd.DataFrame({"object_id": 7, "time_slot": np.arange(64)})
    for condition in ("random_20", "mixed_outage_3"):
        full = delivered_view(values, present, slots, condition, 20261017)
        prefix = delivered_view(values[:27], present[:27], slots.iloc[:27], condition, 20261017)
        for a, b in zip(full, prefix):
            np.testing.assert_array_equal(a[:27], b)
    print(json.dumps({"status": "passed", "checks": ["split_counts", "metric_fixture", "prefix_masks"],
                      "scope": "installation checks, not benchmark performance"}))


if __name__ == "__main__":
    main()
