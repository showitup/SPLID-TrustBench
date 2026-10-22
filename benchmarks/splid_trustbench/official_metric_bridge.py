"""Benchmark-only definitions extracted from benchmarks/splid_trustbench/official_metric_bridge.py."""


from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Iterable
import numpy as np
import pandas as pd


OFFICIAL_COLUMNS = ["ObjectID", "TimeIndex", "Direction", "Node", "Type"]


@dataclass(frozen=True)
class OfficialScore:
    precision: float
    recall: float
    f2: float
    rmse: float
    tp: int
    fp: int
    fn: int
    matched_count: int
    tolerance: int
    object_count: int
    ignored_unknown_object_predictions: int


def validate_official_frame(frame: pd.DataFrame, name: str) -> None:
    missing = [column for column in OFFICIAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing official columns: {missing}")
    if frame[OFFICIAL_COLUMNS].isna().any().any():
        raise ValueError(f"{name} contains null official fields")
    if not frame["Direction"].isin(["EW", "NS", "ES"]).all():
        invalid = sorted(frame.loc[~frame["Direction"].isin(["EW", "NS", "ES"]), "Direction"].unique())
        raise ValueError(f"{name} contains invalid directions: {invalid}")


def score_official_nodes(
    ground_truth: pd.DataFrame,
    participant: pd.DataFrame,
    tolerance: int = 6,
    object_ids: Iterable[int] | None = None,
) -> tuple[OfficialScore, pd.DataFrame, pd.DataFrame]:
    """Score sparse nodes with the released SPLID evaluator's exact semantics.

    Candidate matching is ground-truth ordered and takes the first unmatched
    participant row in the tolerance window. A wrong Node/Type consumes the
    candidate and contributes one FP, matching the public evaluator.
    """

    validate_official_frame(ground_truth, "ground_truth")
    validate_official_frame(participant, "participant")
    gt = ground_truth.copy()
    pred = participant.copy()
    if object_ids is not None:
        allowed = {int(value) for value in object_ids}
        gt = gt[gt["ObjectID"].astype(int).isin(allowed)].copy()
        pred = pred[pred["ObjectID"].astype(int).isin(allowed)].copy()

    gt_object_ids = [int(value) for value in gt["ObjectID"].drop_duplicates().tolist()]
    unknown_mask = ~pred["ObjectID"].astype(int).isin(gt_object_ids)
    ignored_unknown = int(unknown_mask.sum())

    total_tp = total_fp = total_fn = 0
    all_distances: list[float] = []
    object_rows: list[dict] = []
    match_rows: list[dict] = []

    gt_non_es = gt[gt["Direction"] != "ES"]
    pred_non_es = pred[pred["Direction"] != "ES"]
    gt_groups = {int(object_id): frame.copy() for object_id, frame in gt_non_es.groupby("ObjectID", sort=False)}
    pred_groups = {int(object_id): frame.copy() for object_id, frame in pred_non_es.groupby("ObjectID", sort=False)}
    empty_pred = pred_non_es.iloc[0:0].copy()

    for object_id in gt_object_ids:
        gt_obj = gt_groups.get(object_id, gt_non_es.iloc[0:0]).copy()
        pred_obj = pred_groups.get(object_id, empty_pred).copy()
        pred_obj["_matched"] = False
        tp = fp = fn = 0
        distances: list[float] = []

        for gt_index, gt_row in gt_obj.iterrows():
            candidates = pred_obj[
                (pred_obj["TimeIndex"].astype(int) >= int(gt_row["TimeIndex"]) - tolerance)
                & (pred_obj["TimeIndex"].astype(int) <= int(gt_row["TimeIndex"]) + tolerance)
                & (pred_obj["Direction"] == gt_row["Direction"])
                & (~pred_obj["_matched"])
            ]
            if candidates.empty:
                fn += 1
                match_rows.append(
                    {
                        "object_id": object_id,
                        "gt_index": int(gt_index),
                        "pred_index": None,
                        "classification": "FN",
                        "distance": None,
                    }
                )
                continue

            pred_index = candidates.index[0]
            pred_row = candidates.iloc[0]
            distance = int(pred_row["TimeIndex"]) - int(gt_row["TimeIndex"])
            pred_obj.loc[pred_index, "_matched"] = True
            correct = pred_row["Node"] == gt_row["Node"] and pred_row["Type"] == gt_row["Type"]
            if correct:
                tp += 1
                distances.append(distance)
                classification = "TP"
            else:
                fp += 1
                classification = "FP_CLASS"
            match_rows.append(
                {
                    "object_id": object_id,
                    "gt_index": int(gt_index),
                    "pred_index": int(pred_index),
                    "classification": classification,
                    "distance": distance,
                }
            )

        unmatched_indices = pred_obj.index[~pred_obj["_matched"]].tolist()
        fp += len(unmatched_indices)
        for pred_index in unmatched_indices:
            match_rows.append(
                {
                    "object_id": object_id,
                    "gt_index": None,
                    "pred_index": int(pred_index),
                    "classification": "FP_UNMATCHED",
                    "distance": None,
                }
            )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_distances.extend(distances)
        object_precision = tp / (tp + fp) if tp + fp else 0.0
        object_recall = tp / (tp + fn) if tp + fn else 0.0
        object_f2 = (5 * tp) / (5 * tp + 4 * fn + fp) if 5 * tp + 4 * fn + fp else 0.0
        object_rows.append(
            {
                "object_id": object_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": object_precision,
                "recall": object_recall,
                "f2": object_f2,
                "rmse": float(np.sqrt(np.mean(np.square(distances)))) if distances else 0.0,
                "matched_count": len(distances),
            }
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) if 5 * total_tp + 4 * total_fn + total_fp else 0.0
    score = OfficialScore(
        precision=precision,
        recall=recall,
        f2=f2,
        rmse=float(np.sqrt(np.mean(np.square(all_distances)))) if all_distances else 0.0,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        matched_count=len(all_distances),
        tolerance=int(tolerance),
        object_count=len(gt_object_ids),
        ignored_unknown_object_predictions=ignored_unknown,
    )
    return score, pd.DataFrame(object_rows), pd.DataFrame(match_rows)


def score_to_dict(score: OfficialScore) -> dict:
    return asdict(score)
