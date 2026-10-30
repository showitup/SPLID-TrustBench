# Comparing evidence-to-policy methods

SPLID-TrustBench connects three evaluation layers. Each may be used with a
different model: a producer emits orbital evidence, a consumer scores the
delivered view, and a service policy chooses an action. The release contains
evaluation software and contracts; the Orbit-to-Action model implementations
and checkpoints are separate from this benchmark.

## Observation and release

Use the supplied object split for training, validation selection, and evaluation.
The [availability tracks](../benchmarks/splid_trustbench/availability/tracks.json)
declare the producer's information horizon. A retrospective row becomes eligible
after full-trajectory processing. A causal row becomes eligible at its recorded
release time; a consumer admits only rows released by its decision deadline.

For sparse commitments, `bounded_support_6` allows at most six samples between
first candidate support and resolution. Event localization is scored within
the separate ±6-sample tolerance. The evaluator checks submission timestamps
and scores typed nodes; encoder and resolver causality are established through
the accompanying prefix-intervention reports, including candidates resolved
without emission.

## Public evidence and delivery

An alternative producer can export the [OEC schema](../interfaces/orbit_evidence/SCHEMA.md).
All compared policy consumers use the same [admitted fields](../interfaces/orbit_evidence/schemas/policy_evidence_view.json).
Mode posteriors, learned ambiguity, residuals, direction, and validity are public
inputs. Native event targets, direct event/transition channels, and private
hidden states stay outside this policy view.

The [delivery generators](DELIVERY_PROTOCOL.md) support random token deletion,
direction loss, sample gaps, field loss, whole-slot interruptions, and delayed
or asynchronous records. Use identical masks and native-validity inventories
for every method in a comparison. Partial records contain current evidence;
empty records contain none. Retain both in the evaluation inventory.

The causal generator accepts a tensor shaped `(slots, 12, 26)`: each sample
contains 13 policy fields for EW followed by 13 for NS, in the schema's order.
The native-presence array has shape `(slots, 12, 2)`. `delivered_view` returns
masked fields, token presence, an incomplete-record flag, and a nonempty-record
flag. With `return_inventory=True`, it also returns field presence. Object-slot
keys determine the perturbations, so the same prefix receives the same mask
when more observations are appended.

## Targets and policy scoring

Construct targets from native labels before applying delivery loss. For a valid
trajectory of length `T`, enumerate slots `0` through `(T-1)//12`, including its
partial final slot. A slot is risky if any EW or NS native ID, AD, or IK event
falls within its sample interval; SS and ES markers do not define risky slots.
The evaluation partition contains 90,919 slots, with 414 risky and 90,505 normal.
Delivery never changes these targets.

The policy evaluator takes two CSV files:

| File | Required columns |
| --- | --- |
| Ground truth | `object_id,time_slot,risk_label` |
| Submission | `object_id,time_slot,internal_action` |

The submission may also include `public_action` for a mapping check. It must
cover exactly the target object-slot keys. Internal actions map to the service
interface as follows:

| Internal action | Public action | Normal utility | Risky utility |
| --- | --- | ---: | ---: |
| `A` (allow) | `allow` | 1.0 | -2.0 |
| `C` (challenge) | `restrict` | 0.6 | 0.2 |
| `R` (restrict) | `restrict` | -0.4 | 0.6 |
| `D` (deny) | `reject` | -1.0 | 1.0 |

False allowance is the fraction of risky slots receiving `allow`. Normal
restriction is the fraction of normal slots receiving `restrict` or `reject`.
Policy regret is the mean of `1 - utility`, retaining the two internal
restriction severities. Rates are returned as fractions, not percentages.
When a target class is absent, its class-conditional rate is `null`.

Predicted scores, thresholds, and historical state are supplied by the method
being evaluated. The scorer does not fit a model or select an operating point.
The four-slot files under `examples/` illustrate this interface.

## Report a comparable result

Report the observation track, delivery condition, prediction aggregation, and
validation operating point alongside the metric. Compare methods on the same
objects, targets, delivery realizations, and admitted evidence fields. For a
representation control, match training views and readout selection as well.

Pair ranking or classification results with false allowance, normal restriction,
and regret. A common validation service budget tests whether additional risk
protection remains when the allow boundary is selected under the same service
constraint. Report the realized evaluation restriction rate too: matching a
validation budget does not force identical held-out service cost.

Keep retrospective replay and prefix-produced results in distinct comparisons.
For uncertainty intervals, resample complete objects jointly across methods,
preserving their slots and delivery realizations.
