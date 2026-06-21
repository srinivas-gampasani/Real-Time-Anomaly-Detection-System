"""
Ensemble Anomaly Detector
==========================
Combines:
  1. Isolation Forest      — global outlier detection
  2. Autoencoder           — reconstruction-based deep detection
  3. Control Charts        — CUSUM + 3-sigma statistical process control

Fusion strategies:
  - MAJORITY_VOTE (default): flag if ≥2/3 detectors agree
  - ANY: flag if any detector flags (high recall)
  - ALL: flag only if all agree (high precision)
  - WEIGHTED: confidence-weighted score with adaptive threshold

Outputs:
  - Ensemble anomaly flag + confidence
  - Per-detector breakdown
  - Severity score (Low / Medium / High / Critical)
"""

import logging
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

from src.isolation_forest_detector import IsolationForestDetector, DetectionResult
from src.autoencoder_detector import AutoencoderDetector, AutoencoderResult
from src.control_charts import ControlChartsDetector, ControlChartResult

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    MAJORITY_VOTE = "majority_vote"
    ANY           = "any"
    ALL           = "all"
    WEIGHTED      = "weighted"


SEVERITY_THRESHOLDS = {
    "Critical": 0.85,
    "High":     0.65,
    "Medium":   0.40,
    "Low":      0.0
}


@dataclass
class EnsembleResult:
    event_id: str
    timestamp: float
    service: str
    metric_name: str
    value: float
    is_anomaly: bool
    ensemble_confidence: float
    severity: str                    # Low / Medium / High / Critical
    votes: Dict[str, bool]           # per-detector vote
    detector_confidences: Dict[str, float]
    fusion_strategy: str
    latency_ms: float = 0.0

    def to_dict(self):
        return asdict(self)


def _severity(confidence: float) -> str:
    for label, threshold in SEVERITY_THRESHOLDS.items():
        if confidence >= threshold:
            return label
    return "Low"


class EnsembleDetector:
    """
    Ensemble of three detectors with configurable fusion.
    """

    def __init__(
        self,
        fusion_strategy: FusionStrategy = FusionStrategy.MAJORITY_VOTE,
        weights: Dict[str, float] = None,
        if_contamination: float = 0.02,
        ae_latent_dim: int = 8,
        cc_sigma: float = 3.0,
        cc_cusum_h: float = 5.0
    ):
        self.strategy = fusion_strategy
        self.weights = weights or {
            "IsolationForest": 0.35,
            "Autoencoder":     0.35,
            "ControlCharts":   0.30
        }

        self.if_det  = IsolationForestDetector(contamination=if_contamination)
        self.ae_det  = AutoencoderDetector(latent_dim=ae_latent_dim)
        self.cc_det  = ControlChartsDetector(sigma_multiplier=cc_sigma, cusum_h=cc_cusum_h)

        self._fitted = False
        logger.info("EnsembleDetector initialized | strategy=%s | weights=%s",
                    fusion_strategy, self.weights)

    def fit(self, baseline_events: List[dict], ae_epochs: int = 80) -> None:
        """Train all three detectors on baseline normal events."""
        logger.info("Fitting ensemble on %d baseline events...", len(baseline_events))

        baseline_values = [e["value"] for e in baseline_events]
        import numpy as np

        # Isolation Forest
        self.if_det.fit(np.array(baseline_values))

        # Autoencoder
        self.ae_det.fit(baseline_events, epochs=ae_epochs)

        # Control Charts
        self.cc_det.fit(baseline_events)

        self._fitted = True
        logger.info("Ensemble fitted successfully.")

    def _fuse(
        self,
        if_res:  DetectionResult,
        ae_res:  AutoencoderResult,
        cc_res:  ControlChartResult
    ) -> Tuple[bool, float]:
        """
        Fuse three detector outputs into a single anomaly decision.
        Returns (is_anomaly, confidence).
        """
        votes = {
            "IsolationForest": if_res.is_anomaly,
            "Autoencoder":     ae_res.is_anomaly,
            "ControlCharts":   cc_res.is_anomaly,
        }
        confs = {
            "IsolationForest": if_res.confidence,
            "Autoencoder":     ae_res.confidence,
            "ControlCharts":   cc_res.confidence,
        }

        if self.strategy == FusionStrategy.MAJORITY_VOTE:
            n_yes = sum(votes.values())
            is_anomaly = n_yes >= 2
            confidence = sum(confs[k] * self.weights[k] for k in confs) if is_anomaly else \
                         sum(confs[k] * self.weights[k] for k in confs) * 0.5
        elif self.strategy == FusionStrategy.ANY:
            is_anomaly = any(votes.values())
            confidence = max(confs.values())
        elif self.strategy == FusionStrategy.ALL:
            is_anomaly = all(votes.values())
            confidence = min(confs.values())
        else:  # WEIGHTED
            weighted_conf = sum(confs[k] * self.weights[k] for k in confs)
            is_anomaly = weighted_conf > 0.5
            confidence = weighted_conf

        return is_anomaly, round(confidence, 4), votes, confs

    def score_batch(self, events: List[dict]) -> List[EnsembleResult]:
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        t0 = time.time()

        if_results  = self.if_det.score_batch(events)
        ae_results  = self.ae_det.score_batch(events)
        cc_results  = self.cc_det.score_batch(events)

        total_latency_ms = (time.time() - t0) * 1000
        per_event_latency = total_latency_ms / len(events) if events else 0

        results = []
        for i, ev in enumerate(events):
            is_anomaly, confidence, votes, confs = self._fuse(
                if_results[i], ae_results[i], cc_results[i]
            )
            results.append(EnsembleResult(
                event_id=ev["event_id"],
                timestamp=ev["timestamp"],
                service=ev["service"],
                metric_name=ev["metric_name"],
                value=ev["value"],
                is_anomaly=is_anomaly,
                ensemble_confidence=confidence,
                severity=_severity(confidence) if is_anomaly else "Normal",
                votes=votes,
                detector_confidences=confs,
                fusion_strategy=self.strategy,
                latency_ms=round(per_event_latency, 3)
            ))
        return results

    def get_detector_names(self) -> List[str]:
        return ["IsolationForest", "Autoencoder", "ControlCharts"]
