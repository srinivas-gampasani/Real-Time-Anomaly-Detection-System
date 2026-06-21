"""
Isolation Forest Detector
==========================
Unsupervised anomaly detection using Isolation Forest.
Trains on clean baseline data; scores new events in real-time.

Isolation Forest isolates anomalies by randomly partitioning features.
Anomalies require fewer splits → shorter path lengths → higher anomaly score.

Config:
  contamination = 0.02  (expect ~2% anomalies in production streams)
  n_estimators  = 200   (more trees = more stable scores)
  max_samples   = 256   (sub-sampling for speed at 50K eps)
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    event_id: str
    timestamp: float
    service: str
    metric_name: str
    value: float
    score: float           # raw anomaly score (more negative = more anomalous)
    is_anomaly: bool
    confidence: float      # normalised 0-1 anomaly confidence
    detector: str = "IsolationForest"


class IsolationForestDetector:
    """
    Real-time Isolation Forest anomaly detector.
    Fits on baseline window, scores streaming events.
    """

    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 200,
        max_samples: int = 256,
        threshold_percentile: float = 98.0,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.threshold_percentile = threshold_percentile
        self._fitted = False
        self._threshold = None
        logger.info("IsolationForestDetector initialized (contamination=%.2f, n_estimators=%d)",
                    contamination, n_estimators)

    def _features(self, values: np.ndarray) -> np.ndarray:
        """Build feature matrix. values shape: (N,) → (N, 3)"""
        # Features: raw value, z-score proxy (diff from rolling mean), log-transform
        arr = values.reshape(-1, 1)
        log_arr = np.log1p(np.abs(arr))
        diff = np.diff(arr, axis=0, prepend=arr[:1])
        return np.hstack([arr, log_arr, diff])

    def fit(self, baseline_values: np.ndarray) -> None:
        """Train on clean baseline data."""
        X = self._features(baseline_values)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        # Compute threshold from training scores
        train_scores = self.model.decision_function(X_scaled)
        self._threshold = np.percentile(train_scores, 100 - self.threshold_percentile)
        self._fitted = True
        logger.info("IsolationForest fitted on %d samples | threshold=%.4f", len(baseline_values), self._threshold)

    def score_batch(self, events: List[dict]) -> List[DetectionResult]:
        """Score a batch of events, return DetectionResult per event."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        values = np.array([e["value"] for e in events], dtype=np.float64)
        X = self._features(values)
        X_scaled = self.scaler.transform(X)

        raw_scores = self.model.decision_function(X_scaled)
        # Normalise scores to [0, 1] confidence (higher = more anomalous)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            confidence = 1.0 - (raw_scores - min_s) / (max_s - min_s)
        else:
            confidence = np.zeros_like(raw_scores)

        results = []
        for i, ev in enumerate(events):
            is_anomaly = raw_scores[i] < self._threshold
            results.append(DetectionResult(
                event_id=ev["event_id"],
                timestamp=ev["timestamp"],
                service=ev["service"],
                metric_name=ev["metric_name"],
                value=ev["value"],
                score=round(float(raw_scores[i]), 6),
                is_anomaly=is_anomaly,
                confidence=round(float(confidence[i]), 4),
                detector="IsolationForest"
            ))
        return results

    def score_single(self, value: float, event: dict) -> DetectionResult:
        return self.score_batch([{**event, "value": value}])[0]
