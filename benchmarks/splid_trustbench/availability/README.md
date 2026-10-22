# Availability-Aware Evaluation

This directory separates methods by the trajectory information available when a
sparse node is committed. It prevents full-trajectory and future-window methods
from being described as strict online baselines.

Tracks are defined in `tracks.json`:

- `offline_full`
- `lookahead_256`
- `lookahead_32`
- `bounded_6`
- `bounded_support_6` (paper contract)
- `strict_causal`

Every submission contains the official SPLID columns plus `available_at`. Its
metadata must declare the track, lookahead budget, teacher use, causal audit,
and the non-sealed development status. The evaluator fails closed before
computing the official metric.

For causal tracks, the audit must be layered: a raw-trajectory
future-suffix test for the encoder and a cutoff replay test for the commitment
layer must both pass. When the object manifest includes `seq_len`, the evaluator
also rejects every row with `available_at > seq_len - 1`.

The paper's six-step deadline starts at **candidate first support**, not at the
event timestamp. Use `bounded_support_6` and add `first_support_at` to the CSV.
The evaluator checks `available_at - first_support_at <= 6` and reports
event-relative latency separately. Record the actual first support from the
resolver trace; do not reconstruct it from ground-truth event time. Resolution
audits must also include candidates closed without emitting an event, since
these null decisions do not appear in the submission CSV.

`bounded_6` retains the older, stricter event-relative deadline for historical
comparisons. It is not the paper's candidate-resolution guarantee.

Example:

```bash
python benchmarks/splid_trustbench/availability/evaluate_submission.py \
  --ground-truth data/SPLID/phase_2/test_label.csv \
  --submission submission_with_availability.csv \
  --metadata submission.meta.json \
  --object-manifest benchmarks/splid_trustbench/manifests/phase2_objects.csv \
  --output availability_report.json
```

`audit_future_suffix.py` compares public prefix outputs from the same prefix
paired with original and perturbed future suffixes. Causal metadata should cite
the generated report rather than relying only on a method's self-declaration.

The released finalist uses its published pipeline, which mixes 32-step and
256-step future windows. Its score is a released-comparator reference, not a
prefix-only result. Obtain the original SPLID trajectories and labels from the
dataset provider; this artifact distributes protocols and split manifests.
