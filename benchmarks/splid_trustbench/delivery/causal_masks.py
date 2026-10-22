"""Benchmark-only definitions extracted from backend/evaluation/causal_availability.py."""


import numpy as np
import pandas as pd


ROOT_SEEDS = (20261017, 20261018, 20261019)


CONDITIONS = ('clean', 'random_20', 'random_30', 'ew_missing', 'ns_missing',
              'contiguous_4', 'field_missing', 'mixed_outage_3', 'mixed_outage_6')


FIELD_GROUPS = ((0, 1, 2, 3, 7), (4, 6, 8, 10), (5, 9, 11))


def delivered_view(x, presence, slots, condition, seed, return_inventory=False):
    """Masks depend only on object/slot IDs; suffix edits cannot alter a prefix."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    shape = (len(x), x.shape[1], 2, x.shape[-1] // 2)
    field_keep = np.ones(shape, dtype=bool)
    token_keep = np.ones(shape[:-1], dtype=bool)
    if condition != 'clean':
        # Counter-based draws remain identical across cohorts, prefixes and models.
        oid = slots.object_id.to_numpy(np.uint64)
        time = slots.time_slot.to_numpy(np.uint64)
        with np.errstate(over='ignore'):
            base = oid * np.uint64(6364136223846793005) + time * np.uint64(1442695040888963407) + np.uint64(seed)
        def uniform(offset):
            with np.errstate(over='ignore'):
                z = base + np.uint64(offset + 1) * np.uint64(0x9E3779B97F4A7C15)
                z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                z ^= z >> np.uint64(31)
            return (z >> np.uint64(11)).astype(float) / 2**53
        if condition.startswith('random') or condition.startswith('mixed'):
            rate = .2 if condition == 'random_20' else .3 if condition == 'random_30' else .15
            for t in range(x.shape[1]):
                for d in range(2):
                    token_keep[:, t, d] = uniform(2*t+d) >= rate
        if condition in ('ew_missing', 'ns_missing'):
            token_keep[:, :, int(condition == 'ns_missing')] = False
        if condition == 'contiguous_4':
            start = np.floor(uniform(101) * (x.shape[1]-3)).astype(int)
            for t in range(x.shape[1]):
                token_keep[:, t, :] &= ~((t >= start) & (t < start+4))[:, None]
        if condition == 'field_missing' or condition.startswith('mixed'):
            for d in range(2):
                group = np.floor(uniform(201+d) * 3).astype(int)
                drop = np.ones(len(x), bool) if condition == 'field_missing' else uniform(211+d) < .35
                for k, columns in enumerate(FIELD_GROUPS):
                    rows = np.flatnonzero(drop & (group == k))
                    for c in columns:
                        field_keep[rows, :, d, c] = False
        if condition.startswith('mixed'):
            # Intermittent direction-specific failures are independent of risk labels.
            direction_drop, direction = uniform(301) < .25, uniform(302) < .5
            for d in range(2):
                token_keep[direction_drop & (direction == bool(d)), :, d] = False
            length = int(condition.rsplit('_', 1)[1])
            # Fixed 48-slot cycles, selected without object trajectory length.
            start = 8 + (slots.object_id.to_numpy(np.int64) * 17 + seed) % 24
            outage = ((slots.time_slot.to_numpy() % 48) >= start) & ((slots.time_slot.to_numpy() % 48) < start+length)
            token_keep[outage] = False
    observed = presence & token_keep
    raw = x.reshape(shape).copy()
    raw *= field_keep & token_keep[..., None]
    raw = raw.reshape(x.shape)
    incomplete = ((~token_keep & presence).any(axis=(1, 2)) |
                  ((~field_keep) & presence[..., None]).any(axis=(1, 2, 3)))
    result = (raw, observed, incomplete, observed.any(axis=(1, 2)))
    if return_inventory:
        return (*result, field_keep & observed[..., None])
    return result
