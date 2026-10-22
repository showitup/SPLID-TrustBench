# Evidence delivery protocol

## Causal partial-evidence experiments

The generator is `benchmarks/splid_trustbench/delivery/causal_masks.py`.
Masks are counter-based functions of object ID, slot index and seed. Truncating
a cohort or perturbing its future suffix leaves earlier masks unchanged.
The three delivery seeds are 20261017, 20261018 and 20261019.

| Condition | Delivered-view operation |
| --- | --- |
| `clean` | Keep every supplied record; retain original native-validity masks |
| `random_20`, `random_30` | Independently remove 20% or 30% of directional tokens |
| `ew_missing`, `ns_missing` | Remove one direction throughout each evaluated slot |
| `contiguous_4` | Remove both directions over four consecutive samples in a slot |
| `field_missing` | Remove one public field group per direction; preserve structural validity |
| `mixed_outage_3`, `mixed_outage_6` | Combine partial loss with entire-slot interruptions |

Mixed loss uses 15% token deletion, a field-group drop probability of 0.35 per
direction, and a 0.25 probability of losing one direction. Whole-slot outages
last three or six slots in fixed 48-slot cycles, with object/seed-specific start
positions. Mask generation reads neither targets nor trajectory length.

The field groups use the exact order in `policy_evidence_view.json`. Native
invalid observations and partial final slots are accounted for before delivery
loss. In local-ranking evaluation, unavailable slots receive the fixed score
zero; policy evaluation instead uses finite historical support.

## Delayed or asynchronous intact evidence

The retrospective transport experiments use
`benchmarks/splid_trustbench/delivery/transport.py`, with seed 20260809. They retain
the evidence content and vary arrival time. Uniform delays are one, three or
six decision slots; asynchronous delays differ by public field group.
The generator represents intact records arriving late; it does not implement
the paper's timestamp-aware recovery method.

One raw SPLID sample is two hours; one full decision slot contains 12 samples.
A waiting budget is therefore measured in decision slots, separately from
candidate-resolution delay, which is measured in raw samples. Controlled token
loss and retrospective timestamp recovery are distinct experiments, not one
continuous accuracy ladder.
