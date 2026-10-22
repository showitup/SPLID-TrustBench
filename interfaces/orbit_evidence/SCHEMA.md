# Orbit Evidence Schema

Current version: `1.1.0`

## `orbit_evidence.parquet`

Primary key:

- `object_id`
- `time_idx`
- `direction`

Required columns:

- `object_id`
- `time_idx`
- `direction`
- `timestamp_utc`
- `split`
- `p_background`
- `p_start`
- `p_inside`
- `p_end`
- `p_id`
- `p_ad`
- `p_ik`
- `p_mode_nk`
- `p_mode_ck`
- `p_mode_ek`
- `p_mode_hk`
- `behavior_consistency`
- `forecast_residual`
- `physics_residual`
- `uncertainty`
- `event_logit_margin`
- `mode_entropy`
- `is_valid`

Additional required columns for `schema_version = 1.1.0`:

- `drift_residual`
- `physics_violation`
- `drift_risk`
- `physics_risk`
- `evidence_schema_version`

Optional public evidence columns:

- `evidence_reliability`: producer-supplied reliability output in `[0, 1]`.
  This field is allowed in `orbit_evidence.parquet` but is not a required v1.1
  field. Back-end modules may consume it only after the public adapter schema is
  explicitly upgraded to aggregate it.
- `availability_mode`: declares online bounded-latency or full-trajectory
  retrospective availability.
- `available_at_time_idx`, `predicted_event_time_idx`, `predicted_event_lag`,
  `tuple_existence_probability`, and `predicted_duration`: public structured
  tuple evidence; none may expose hidden states.
- `committed_event`, `committed_event_score`,
  `committed_available_at_time_idx`, `committed_event_time_idx`,
  `committed_node`, `committed_following_mode`, and
  `committed_latency_steps`: optional single-commit fields. This latency is
  measured from the reported event time, separately from candidate resolution.
- `committed_first_support_time_idx`: required when metadata declares
  `availability_contract.resolution_origin = "first_support"`. The paper's
  bounded-delay track requires release no more than six samples after this
  first-support timestamp; event-relative latency may exceed six samples.

## Value Rules

- `direction ∈ {EW, NS}`
- `split ∈ {train, val, test, stress_train, stress_val, stress_test}`
- `is_valid ∈ {0, 1}`
- key columns must never be null
- valid rows must have no nulls in required score fields
- invalid rows may have null score fields, but must keep the row
- any row with null score fields must have `is_valid = 0`
- probability fields must be in `[0, 1]` when non-null
- `p_background + p_start + p_inside + p_end = 1 ± 1e-4` on valid rows
- `p_id + p_ad + p_ik = 1 ± 1e-4` on valid rows
- `p_mode_nk + p_mode_ck + p_mode_ek + p_mode_hk = 1 ± 1e-4` on valid rows
- `behavior_consistency ∈ [0, 1]`
- `uncertainty ∈ [0, 1]`
- `forecast_residual >= 0`
- `physics_residual >= 0`
- `mode_entropy >= 0`
- `drift_residual >= 0` for v1.1
- `physics_violation >= 0` for v1.1
- `drift_risk ∈ [0, 1]` for v1.1
- `physics_risk ∈ [0, 1]` for v1.1
- `evidence_reliability ∈ [0, 1]` when present

- Optional commitment fields are public decoder outputs, not labels or private tensors.
- Commitment fields are atomic: if one is present, all seven must be present.
- For `committed_event = 1`, `committed_event_time_idx = time_idx`,
  `committed_latency_steps = committed_available_at_time_idx - time_idx`, and
  latency must be non-negative. With the first-support contract, the resolution
  budget applies to `committed_available_at_time_idx - committed_first_support_time_idx`.
  Historical metadata using `online_maximum_lookahead` alone retains its older
  event-relative latency check.
- A producer must not overwrite event-time causal probability fields with a
  commitment that becomes available later. Consumers must gate optional
  commitment fields by `committed_available_at_time_idx`.

## Version Semantics

- `1.0.0`: event/mode/transition/uncertainty/consistency evidence.
- `1.1.0`: adds drift-sensitive and physics-consistency evidence.
- Optional reliability and commitment columns may be present, but do not change the required
  v1.1 contract until the adapter schema is versioned.

The validator remains backward-compatible with v1.0 files. Back-end modules must
consume v1.1 only through the public policy view; they must not read producer hidden
states or logits.

`orbit_evidence` v1.1 remains the common contract for both dense causal evidence
and optional committed tuples. Optional fields do not change the primary key or
permit a backend to read frontend hidden state. Producers must record the
availability mode and maximum lookahead in metadata.

## Coverage Rules

When validated against `benchmarks/splid_trustbench/manifests/phase2_objects.csv`:

- only `benchmark_valid = 1` objects are required
- every valid object must have exactly `seq_len(object_id) * 2` rows
- every valid `(object_id, direction)` stream must have exactly `seq_len(object_id)` rows
- malformed objects marked `benchmark_valid = 0` must not appear unless explicitly validating a diagnostic file

## `orbit_evidence.meta.json`

Required fields:

- `schema_version`
- `producer_name`
- `producer_version`
- `source_split_manifest`
- `label_protocol_version`
- `feature_spec_version`
- `decoder_config`
- `calibration_config`
- `normalization_stats_source`
- `created_at`
- `evidence_file`
- `row_count`
- `object_count`
- `invalid_row_policy`
- `invalid_rows`

The metadata must be committed or archived with each evidence file. Back-end results must record the metadata path or a content hash.
