# SPLID-TrustBench

Benchmark protocols and evaluation tools for availability-conditioned satellite
pattern-of-life inference and public orbit evidence.

## Installation

Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m benchmarks.splid_trustbench.artifact_smoke
```

## Data and protocols

Obtain trajectories and annotations from the original SPLID provider. The
object-disjoint partitions contain 1,520 training, 380 validation, and 498
evaluation objects. Raw data are not included.

| Component | Documentation or entry point |
| --- | --- |
| Data and object partitions | [Data card](docs/DATA_CARD.md) |
| Sparse-node and availability evaluation | [Evaluation guide](benchmarks/splid_trustbench/availability/README.md) |
| Public Orbit-Evidence Contract | [OEC schema](interfaces/orbit_evidence/SCHEMA.md) |
| Admitted policy features | [Policy evidence view](interfaces/orbit_evidence/schemas/policy_evidence_view.json) |
| Missing, delayed, and asynchronous evidence | [Delivery protocol](docs/DELIVERY_PROTOCOL.md) |

## Evaluate a submission

```bash
python benchmarks/splid_trustbench/availability/evaluate_submission.py \
  --ground-truth path/to/test_label.csv \
  --submission submission.csv \
  --metadata examples/offline_submission.meta.json \
  --object-manifest benchmarks/splid_trustbench/manifests/phase2_objects.csv \
  --output outputs/submission_report.json
```

The submission columns are
`ObjectID,TimeIndex,Direction,Node,Type,available_at`.
The retrospective metadata template above can be copied and edited for the
submitted method. The `bounded_support_6` track additionally requires
`first_support_at` and causal-audit metadata. Its six-step resolution budget
starts at candidate first support; the six-step event-matching tolerance is
evaluated separately.

## License and citation

Benchmark code and documentation are provided under the [MIT License](LICENSE).
Original datasets retain their providers' terms. Citation metadata are in
[CITATION.cff](CITATION.cff).
