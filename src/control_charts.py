"""
Statistical Control Charts Detector
======================================
Implements two classical SPC (Statistical Process Control) methods:

1. 3-Sigma (Shewhart) Control Charts
   - Flag values beyond μ ± 3σ
   - Fast, interpretable, zero false-positive guarantee at normal dist.

2. CUSUM (Cumulative Sum Control Chart)
   - Detects sustained shifts away from the process mean
   - More sensitive to gradual drifts (e.g. memory leaks)
   - Uses two one-sided CUSUM statistics: C+ (upward) and C- (downward)

3. Adaptive Thresholds
   - Rolling window recalculates μ and σ every N events
   - Adapts to concept drift without full retraining

Reference: Montgomery, D.C. "Introduction to Statistical Quality Control"
"""

import logging
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ControlChartResult:
    event_id: str
    timestamp: float
    service: str
    metric_name: str
    value: float
    z_score: float
    cusum_pos: float
    cusum_neg: float
    control_mean: float
    control_std: float
    upper_limit: float        # UCL = μ + 3σ
    lower_limit: float        # LCL = μ - 3σ
    is_anomaly: bool
    rule_triggered: str       # "3sigma" | "cusum_up" | "cusum_down" | ""
    confidence: float
    detector: str = "ControlCharts"


class AdaptiveControlChart:
    """
    Per-metric adaptive control chart with 3-sigma and CUSUM.
    Maintains a rolling baseline window for adaptive threshold computation.
    """

    def __init__(
        self,
        metric_name: str,
        baseline_window: int = 500,
        sigma_multiplier: float = 3.0,
        cusum_k: float = 0.5,       # slack parameter (allowable slack)
        cusum_h: float = 5.0,       # decision interval
        adaptive: bool = True
    ):
        self.metric_name = metric_name
        self.baseline_window = baseline_window
        self.sigma_multiplier = sigma_multiplier
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.adaptive = adaptive

        self._window: deque = deque(maxlen=baseline_window)
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._fitted_mean: Optional[float] = None
        self._fitted_std:  Optional[float] = None

    def _get_stats(self) -> Tuple[float, float]:
        if self.adaptive and len(self._window) >= 30:
            arr = np.array(self._window)
            return float(arr.mean()), float(arr.std()) + 1e-8
        elif self._fitted_mean is not None:
            return self._fitted_mean, self._fitted_std
        elif len(self._window) > 0:
            arr = np.array(self._window)
            return float(arr.mean()), float(arr.std()) + 1e-8
        return 0.0, 1.0

    def fit(self, baseline_values: np.ndarray) -> None:
        """Pre-warm with baseline normal data."""
        for v in baseline_values:
            self._window.append(v)
        self._fitted_mean = float(baseline_values.mean())
        self._fitted_std  = float(baseline_values.std()) + 1e-8
        logger.info("ControlChart[%s] fitted: μ=%.4f σ=%.4f",
                    self.metric_name, self._fitted_mean, self._fitted_std)

    def update(self, value: float, event: dict) -> ControlChartResult:
        """Process one value, update charts, return detection result."""
        mean, std = self._get_stats()

        # 3-sigma bounds
        ucl = mean + self.sigma_multiplier * std
        lcl = mean - self.sigma_multiplier * std

        # Z-score
        z = (value - mean) / std

        # CUSUM update (standardised)
        k = self.cusum_k
        self._cusum_pos = max(0.0, self._cusum_pos + (z - k))
        self._cusum_neg = max(0.0, self._cusum_neg + (-z - k))

        # Rule checks
        rule = ""
        if abs(z) > self.sigma_multiplier:
            rule = "3sigma"
        elif self._cusum_pos > self.cusum_h:
            rule = "cusum_up"
        elif self._cusum_neg > self.cusum_h:
            rule = "cusum_down"

        is_anomaly = rule != ""

        # Confidence: blend z-score magnitude and CUSUM magnitude
        z_conf     = min(abs(z) / (self.sigma_multiplier * 2), 1.0)
        cusum_conf = min(max(self._cusum_pos, self._cusum_neg) / (self.cusum_h * 2), 1.0)
        confidence = round(max(z_conf, cusum_conf), 4)

        # Update rolling window (only non-anomalous for adaptive)
        if not is_anomaly or not self.adaptive:
            self._window.append(value)

        # Reset CUSUM after detection
        if rule in ("cusum_up", "cusum_down"):
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0

        return ControlChartResult(
            event_id=event["event_id"],
            timestamp=event["timestamp"],
            service=event["service"],
            metric_name=event["metric_name"],
            value=value,
            z_score=round(z, 4),
            cusum_pos=round(self._cusum_pos, 4),
            cusum_neg=round(self._cusum_neg, 4),
            control_mean=round(mean, 4),
            control_std=round(std, 4),
            upper_limit=round(ucl, 4),
            lower_limit=round(lcl, 4),
            is_anomaly=is_anomaly,
            rule_triggered=rule,
            confidence=confidence
        )


class ControlChartsDetector:
    """
    Multi-metric, multi-service adaptive control chart detector.
    Maintains per-metric chart instances.
    """

    def __init__(
        self,
        sigma_multiplier: float = 3.0,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        baseline_window: int = 500,
        adaptive: bool = True
    ):
        self.sigma_multiplier = sigma_multiplier
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.baseline_window = baseline_window
        self.adaptive = adaptive
        self._charts: Dict[str, AdaptiveControlChart] = {}
        logger.info("ControlChartsDetector initialized (sigma=%.1f, CUSUM h=%.1f)",
                    sigma_multiplier, cusum_h)

    def _get_chart(self, metric_name: str) -> AdaptiveControlChart:
        if metric_name not in self._charts:
            self._charts[metric_name] = AdaptiveControlChart(
                metric_name=metric_name,
                baseline_window=self.baseline_window,
                sigma_multiplier=self.sigma_multiplier,
                cusum_k=self.cusum_k,
                cusum_h=self.cusum_h,
                adaptive=self.adaptive
            )
        return self._charts[metric_name]

    def fit(self, events: List[dict]) -> None:
        """Pre-warm all charts with baseline events."""
        by_metric: Dict[str, List[float]] = {}
        for ev in events:
            by_metric.setdefault(ev["metric_name"], []).append(ev["value"])
        for metric, vals in by_metric.items():
            self._get_chart(metric).fit(np.array(vals))
        logger.info("ControlCharts fitted for metrics: %s", list(by_metric.keys()))

    def score_batch(self, events: List[dict]) -> List[ControlChartResult]:
        results = []
        for ev in events:
            chart  = self._get_chart(ev["metric_name"])
            result = chart.update(ev["value"], ev)
            results.append(result)
        return results
