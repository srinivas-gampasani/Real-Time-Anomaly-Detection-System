"""
Anomaly Detection Evaluation
==============================
Computes:
  - Precision, Recall, F1, Accuracy
  - False Positive Rate (FPR)
  - Detection latency (time from anomaly start to first detection)
  - Per-detector breakdown
  - Per-anomaly-type breakdown

Matches against ground truth labels from the event simulator.
"""

import logging
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    method: str
    total_events: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    false_positive_rate: float
    avg_confidence_anomaly: float = 0.0
    avg_confidence_normal:  float = 0.0
    per_type: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def compute_metrics(
    method: str,
    predictions: List[bool],
    ground_truth: List[bool],
    confidences: List[float] = None,
    anomaly_types: List[Optional[str]] = None
) -> EvalMetrics:
    assert len(predictions) == len(ground_truth), "Length mismatch"

    TP = sum(p and g for p, g in zip(predictions, ground_truth))
    FP = sum(p and not g for p, g in zip(predictions, ground_truth))
    TN = sum(not p and not g for p, g in zip(predictions, ground_truth))
    FN = sum(not p and g for p, g in zip(predictions, ground_truth))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / len(predictions) if predictions else 0.0
    fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    avg_conf_anom   = 0.0
    avg_conf_normal = 0.0
    if confidences:
        anom_confs   = [c for c, g in zip(confidences, ground_truth) if g]
        normal_confs = [c for c, g in zip(confidences, ground_truth) if not g]
        avg_conf_anom   = float(np.mean(anom_confs))   if anom_confs   else 0.0
        avg_conf_normal = float(np.mean(normal_confs)) if normal_confs else 0.0

    # Per anomaly type breakdown
    per_type = {}
    if anomaly_types:
        types = set(t for t in anomaly_types if t)
        for atype in types:
            idx  = [i for i, t in enumerate(anomaly_types) if t == atype]
            tp_t = sum(predictions[i] and ground_truth[i] for i in idx)
            fn_t = sum(not predictions[i] and ground_truth[i] for i in idx)
            rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
            per_type[atype] = {
                "count": len(idx),
                "detected": tp_t,
                "recall": round(rec_t, 4)
            }

    return EvalMetrics(
        method=method,
        total_events=len(predictions),
        true_positives=TP,
        false_positives=FP,
        true_negatives=TN,
        false_negatives=FN,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        accuracy=round(accuracy, 4),
        false_positive_rate=round(fpr, 4),
        avg_confidence_anomaly=round(avg_conf_anom, 4),
        avg_confidence_normal=round(avg_conf_normal, 4),
        per_type=per_type
    )


def compute_detection_lag(
    events: List[dict],
    predictions: List[bool],
    anomaly_start_offsets: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute time from anomaly injection to first detection per scenario.
    anomaly_start_offsets: {scenario_name: start_timestamp_ms}
    Returns: {scenario_name: lag_seconds}
    """
    first_detection: Dict[str, float] = {}
    for ev, pred in zip(events, predictions):
        if not pred or ev.get("anomaly_type") is None:
            continue
        atype = ev["anomaly_type"]
        ts    = ev["timestamp"]
        if atype not in first_detection:
            first_detection[atype] = ts

    lags = {}
    for atype, start_ms in anomaly_start_offsets.items():
        if atype in first_detection:
            lags[atype] = round((first_detection[atype] - start_ms) / 1000, 2)
        else:
            lags[atype] = None
    return lags


def print_eval_table(results: Dict[str, EvalMetrics]):
    print("\n" + "=" * 80)
    print("  ANOMALY DETECTION EVALUATION REPORT")
    print("=" * 80)
    print(f"  {'Method':<22} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Acc':>7} {'FPR':>7}")
    print("-" * 80)
    for method, m in results.items():
        print(f"  {method:<22} {m.precision:>7.4f} {m.recall:>7.4f} {m.f1:>7.4f} "
              f"{m.accuracy:>7.4f} {m.false_positive_rate:>7.4f}")
    print("=" * 80)
