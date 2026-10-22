"""Benchmark-only definitions extracted from backend/evaluation/evidence_degradation.py."""


from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd


TYPE_NAMES = ("mode", "dynamics", "physics")


TYPE_FEATURE_INDICES = {
    "mode": (0, 1, 2, 3, 7),
    "dynamics": (4, 6, 8, 10),
    "physics": (5, 9, 11, 12),
}


@dataclass(frozen=True)
class DegradationCondition:
    name: str
    family: str
    level: str
    value: float | int | tuple[int, int, int]


@dataclass
class DegradedEvidence:
    values: np.ndarray  # [slot, direction-step, type, feature]
    presence: np.ndarray  # [slot, direction-step, type]
    availability: np.ndarray  # [slot, direction-step, type]
    age: np.ndarray  # [slot, direction-step, type]
    outage_mask: np.ndarray  # [slot]


def fixed_conditions() -> list[DegradationCondition]:
    conditions = [DegradationCondition("clean", "clean", "clean", 0.0)]
    conditions.extend(
        DegradationCondition(f"token_missing_{int(rate * 100):02d}", "token_missing", f"{int(rate * 100)}%", rate)
        for rate in (0.10, 0.20, 0.30, 0.50)
    )
    conditions.extend(
        DegradationCondition(f"slot_outage_{length:02d}", "slot_outage", f"{length}", length)
        for length in (1, 3, 6, 12)
    )
    conditions.extend(
        DegradationCondition(f"delay_{delay:02d}", "delay", f"{delay}", delay)
        for delay in (1, 3, 6)
    )
    conditions.append(DegradationCondition("async_event1_mode3_physics6", "asynchronous", "1/3/6", (3, 1, 6)))
    conditions.extend(
        DegradationCondition(f"noise_{name}", "noise", name, sigma)
        for name, sigma in (("low", 0.05), ("medium", 0.15), ("high", 0.30))
    )
    return conditions


def directional_tokens(x: np.ndarray, token_presence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert [slot, step, EW+NS features] into 24 direction-step tokens."""
    if x.ndim != 3 or x.shape[-1] % 2:
        raise ValueError("Expected concatenated EW/NS evidence features")
    half = x.shape[-1] // 2
    values = np.concatenate([x[..., :half], x[..., half:]], axis=1).astype(np.float32)
    presence = np.concatenate([token_presence[..., 0], token_presence[..., 1]], axis=1).astype(bool)
    return values, presence


def typed_clean(values: np.ndarray, presence: np.ndarray) -> DegradedEvidence:
    masks = np.zeros((len(TYPE_NAMES), values.shape[-1]), dtype=np.float32)
    for type_index, name in enumerate(TYPE_NAMES):
        masks[type_index, list(TYPE_FEATURE_INDICES[name])] = 1.0
    typed = values[:, :, None, :] * masks[None, None, :, :]
    typed_presence = np.repeat(presence[:, :, None], len(TYPE_NAMES), axis=2)
    return DegradedEvidence(
        values=typed.astype(np.float32),
        presence=typed_presence.copy(),
        # Delivery state is independent of native validity. A clean but invalid
        # token was still delivered and carries an explicit is_valid value.
        availability=np.ones(typed_presence.shape, dtype=bool),
        age=np.zeros(typed_presence.shape, dtype=np.float32),
        outage_mask=np.zeros(len(values), dtype=bool),
    )


def _stable_uint64(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _object_groups(slots: pd.DataFrame) -> Iterable[np.ndarray]:
    for _, indices in slots.groupby("object_id", sort=True).indices.items():
        ordered = np.asarray(indices, dtype=np.int64)
        order = np.argsort(slots.iloc[ordered]["time_slot"].to_numpy(), kind="mergesort")
        yield ordered[order]


def _copy_from_source(result: DegradedEvidence, clean: DegradedEvidence, target: int, source: int, types: tuple[int, ...]) -> None:
    result.values[target, :, types, :] = clean.values[source, :, types, :]
    result.presence[target, :, types] = clean.presence[source, :, types]


def apply_condition(
    values: np.ndarray,
    presence: np.ndarray,
    slots: pd.DataFrame,
    condition: DegradationCondition,
    seed: int,
) -> DegradedEvidence:
    clean = typed_clean(values, presence)
    result = DegradedEvidence(
        values=clean.values.copy(),
        presence=clean.presence.copy(),
        availability=clean.availability.copy(),
        age=clean.age.copy(),
        outage_mask=clean.outage_mask.copy(),
    )
    if condition.family == "clean":
        return result

    if condition.family == "token_missing":
        rate = float(condition.value)
        rng = np.random.default_rng(_stable_uint64(condition.name, seed))
        keep = rng.random(presence.shape) >= rate
        result.presence &= keep[:, :, None]
        result.availability &= keep[:, :, None]
        result.values *= result.presence[..., None]
        return result

    if condition.family == "noise":
        sigma = float(condition.value)
        rng = np.random.default_rng(_stable_uint64(condition.name, seed))
        noise = rng.normal(0.0, sigma, size=result.values.shape).astype(np.float32)
        result.values += noise * result.presence[..., None]
        return result

    if condition.family == "delay":
        delay = int(condition.value)
        for indices in _object_groups(slots):
            for position, target in enumerate(indices):
                source_position = position - delay
                if source_position < 0:
                    result.values[target] = 0.0
                    result.presence[target] = False
                    result.availability[target] = False
                else:
                    source = int(indices[source_position])
                    _copy_from_source(result, clean, int(target), source, (0, 1, 2))
                    result.availability[target] = True
                    result.age[target] = float(delay)
        return result

    if condition.family == "asynchronous":
        delays = tuple(int(item) for item in condition.value)
        for indices in _object_groups(slots):
            for position, target in enumerate(indices):
                for type_index, delay in enumerate(delays):
                    source_position = position - delay
                    if source_position < 0:
                        result.values[target, :, type_index] = 0.0
                        result.presence[target, :, type_index] = False
                        result.availability[target, :, type_index] = False
                    else:
                        source = int(indices[source_position])
                        _copy_from_source(result, clean, int(target), source, (type_index,))
                        result.availability[target, :, type_index] = True
                        result.age[target, :, type_index] = float(delay)
        return result

    if condition.family == "slot_outage":
        length = int(condition.value)
        for indices in _object_groups(slots):
            if len(indices) < 2:
                continue
            usable = min(length, max(len(indices) - 1, 1))
            max_start = max(len(indices) - usable, 1)
            start = 1 + (_stable_uint64(slots.iloc[indices[0]]["object_id"], condition.name, seed) % max_start)
            start = min(int(start), len(indices) - usable)
            source_position = start - 1
            source = int(indices[source_position]) if source_position >= 0 else -1
            for offset in range(usable):
                target = int(indices[start + offset])
                result.outage_mask[target] = True
                result.availability[target] = False
                if source >= 0:
                    _copy_from_source(result, clean, target, source, (0, 1, 2))
                    result.age[target] = float(offset + 1)
                else:
                    result.values[target] = 0.0
                    result.presence[target] = False
        return result

    raise ValueError(f"Unknown degradation family: {condition.family}")


def collapse_for_control(degraded: DegradedEvidence) -> tuple[np.ndarray, np.ndarray]:
    """Collapse disjoint typed channels back to original direction-step tokens."""
    values = degraded.values.sum(axis=2).astype(np.float32)
    presence = degraded.presence.any(axis=2)
    values *= presence[..., None]
    return values, presence
