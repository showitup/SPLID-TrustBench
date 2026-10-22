# SPLID-TrustBench

Availability-conditioned orbital evidence for satellite IoT privilege review.

The benchmark separates what an orbital producer has observed from what a
downstream service has received. It evaluates retrospective reconstruction,
prefix-only commitments, and public-evidence consumption under complete,
partial, and unavailable records.

Repository: <https://github.com/showitup/SPLID-TrustBench>.

## Contents

| Component | Entry |
| --- | --- |
| Official sparse-node metric | `benchmarks/splid_trustbench/official_metric_bridge.py` |
| Availability-aware evaluator | `benchmarks/splid_trustbench/availability/evaluate_submission.py` |
| Object-disjoint partitions | `benchmarks/splid_trustbench/manifests/phase2_objects.csv` |
| Public OEC and policy view | `interfaces/orbit_evidence/schemas/` |
| Causal missing-evidence generator | `benchmarks/splid_trustbench/delivery/causal_masks.py` |
| Delayed and asynchronous delivery | `benchmarks/splid_trustbench/delivery/transport.py` |

This repository contains benchmark protocols and evaluation utilities. Model
architectures, training, recovery consumers, policy-memory implementations,
checkpoints, and raw datasets are not included. The SPLID risk target is a
behavior-event proxy derived from ID/AD/IK nodes.

## Installation and checks

Use Python 3.10 or later, from the extracted repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m benchmarks.splid_trustbench.artifact_smoke
```

The smoke command checks evaluator installation, fixed splits, and deterministic
prefix-preserving delivery masks. Its constructed examples are format checks,
not experimental results.

## Evaluate a sparse-node submission

Obtain SPLID data from its original providers, following `docs/DATA_CARD.md`.
The release retains the 1,520/380/498 object split and omits native labels from
the public split table. SS initial states are scored; ES end markers are not.
The official metric uses one-to-one, ground-truth-ordered matching within six
samples. A wrong Node/Type consumes the candidate and counts as a false positive,
following the released SPLID metric.

```bash
python benchmarks/splid_trustbench/availability/evaluate_submission.py \
  --ground-truth /path/to/SPLID/test_label.csv \
  --submission submission.csv \
  --metadata submission.meta.json \
  --object-manifest benchmarks/splid_trustbench/manifests/phase2_objects.csv \
  --output outputs/submission_report.json
```

`submission.csv` has `ObjectID,TimeIndex,Direction,Node,Type,available_at`.
For the paper's `bounded_support_6` track, it also contains `first_support_at`.
See `benchmarks/splid_trustbench/availability/README.md` for metadata and audits.
`examples/offline_submission.meta.json` provides a retrospective metadata template.
Six-step localization tolerance and six-step candidate resolution are separate
quantities. The historical `bounded_6` event-relative track remains available.

## Consume public evidence

The row schema is `orbit_evidence.schema.json`; the admitted downstream feature
order is `policy_evidence_view.json`. A direction has 13 policy features; paired
slots have shape `[slots, 12, 26]`. Presence has shape `[slots, 12, 2]` and refers
to native validity before transport loss. Object and slot IDs select masks but
are not learned risk features.

```python
from benchmarks.splid_trustbench.delivery.causal_masks import delivered_view

damaged, observed, incomplete, available = delivered_view(
    paired_values, native_presence, slot_index, "random_20", seed=20261017
)
```

`slot_index` contains `object_id` and `time_slot`. Partially observed records
remain consumable; `available == False` identifies entire-slot unavailability.
See `docs/DELIVERY_PROTOCOL.md` for failure composition, seeds and delay units.

Validate a producer export with:

```bash
python interfaces/orbit_evidence/validate_orbit_evidence.py \
  --evidence orbit_evidence.parquet --meta orbit_evidence.meta.json \
  --object-manifest benchmarks/splid_trustbench/manifests/phase2_objects.csv \
  --output-report outputs/evidence_validation.json
```

## Scope

The utilities score supplied predictions and construct reproducible delivery
conditions. They do not train a model, choose a policy, or reproduce the paper's
trained-method results without those separate implementations.

## Citation, licensing and release status

GitHub can read the accompanying `CITATION.cff`. Publication-author metadata
will be updated when finalized. No DOI
or tagged release is claimed here. Code licensing is awaiting the owners'
choice; no reuse license is granted here. Raw datasets and third-party
implementations retain their original terms and are not bundled.
