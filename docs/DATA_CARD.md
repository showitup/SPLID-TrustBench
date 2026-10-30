# Data and partitions

SPLID-TrustBench uses the SPLID Phase 2 orbital pattern-of-life data described in
the [official dataset documentation](https://splid-devkit.readthedocs.io/en/latest/dataset.html).
Obtain trajectories and native labels through the original provider. They are
not redistributed in this protocol artifact.

The snapshot contains 2,401 object trajectories; 2,398 pass parsing. The supplied
manifest retains invalid-object flags and the fixed 1,520/380/498 object-disjoint
train/validation/evaluation allocation. It contains object IDs, lengths and split
metadata, not orbital samples or native event annotations. The split table is
reduced to assignment fields, excluding label-derived stratification features.

Each raw sample represents two hours. Downstream decisions group 12 samples and
retain final partial slots. There are 90,919 evaluation slots, including 337
partial tails. Native ID/AD/IK events define 414 positive slots; the other 90,505
are negative. Targets are built from the original annotations, never from the
delivered or perturbed view.

The reported 498-object cohort is the public SPLID evaluation cohort, not a new
sealed test dataset. All fitting and parameter selection use the fixed training
and validation partitions. The paper distinguishes complete retrospective
evidence, causal damaged evidence and matched readout conditions.

MAD-LEO provides independent orbital validation in the paper. OPPORTUNITY and
OPS-SAT assess decoding and aggregation on other tasks. These datasets and
their trained models are obtained separately from this SPLID protocol bundle.
