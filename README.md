# SPLID-TrustBench

SPLID-TrustBench evaluates satellite pattern-of-life inference from observation
to service decision. It makes three comparison conditions explicit: what a
monitor could observe, which evidence reached the consumer, and what service
cost its decisions incurred.

The benchmark extends the original SPLID task with availability contracts,
a public Orbit-Evidence Contract (OEC), and reproducible delivery failures.
New producers and consumers can be evaluated through these shared interfaces.

| Research question | Reusable resource |
| --- | --- |
| How much does future context contribute? | Availability tracks and sparse-node scoring |
| Can another consumer use the evidence? | OEC schemas and the admitted policy view |
| What survives token, channel, or record loss? | Deterministic delivery generators |
| What protection and service cost result? | Policy-action evaluator and fixed utility |

## Installation

Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m benchmarks.splid_trustbench.artifact_smoke
```

The installation check uses illustrative records and verifies the bundled
split counts, scoring, and prefix-consistent delivery masks.

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
| Service actions and utility | [Policy scoring and comparison guide](docs/BENCHMARK_GUIDE.md) |

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

## Score service decisions

```bash
python -m benchmarks.splid_trustbench.evaluate_policy \
  --ground-truth examples/policy_targets.csv \
  --submission examples/policy_submission.csv \
  --output outputs/policy_report.json
```

This four-slot example returns false allowance `0.5`, normal restriction `0.5`,
and policy regret `0.85`. Replace it with predictions for the full evaluation
slot inventory, including interrupted records. The evaluator joins by object
and slot, applies the public action mapping, and scores the fixed utility.
See the [comparison guide](docs/BENCHMARK_GUIDE.md) for target construction,
availability conditions, and service-budget reporting.

## License and citation

Benchmark code and documentation are provided under the [MIT License](LICENSE).
Original datasets retain their providers' terms. Citation metadata are in
[CITATION.cff](CITATION.cff).
