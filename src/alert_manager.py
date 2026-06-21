"""
Alert Manager + Delta Lake Writer
===================================
Handles downstream actions on confirmed anomalies:

1. Alert routing:
   - Critical / High → PagerDuty (simulated: writes to alerts/pagerduty/)
   - Medium         → Slack     (simulated: writes to alerts/slack/)
   - Low            → Log only

2. Delta Lake audit table:
   - All flagged anomalies written as Parquet-format JSON to delta_lake/
   - Partitioned by date/service for efficient querying

3. Alert deduplication:
   - 60-second cooldown per (service, metric) to suppress alert storms
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from src.ensemble_detector import EnsembleResult

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    alert_id: str
    triggered_at: str
    severity: str
    service: str
    metric_name: str
    value: float
    confidence: float
    channel: str           # "pagerduty" | "slack" | "log"
    message: str
    event_id: str
    acknowledged: bool = False

    def to_dict(self):
        return asdict(self)


class AlertManager:
    """
    Manages alert routing, deduplication, and Delta Lake persistence.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        cooldown_sec: float = 60.0,
        pagerduty_severities: List[str] = None,
        slack_severities: List[str] = None
    ):
        self.output_dir     = Path(output_dir)
        self.cooldown_sec   = cooldown_sec
        self.pd_severities  = set(pagerduty_severities or ["Critical", "High"])
        self.sl_severities  = set(slack_severities    or ["Medium"])
        self._last_alert:   Dict[str, float] = {}   # key: (service, metric) → last alert ts

        # Create output dirs
        (self.output_dir / "alerts" / "pagerduty").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "alerts" / "slack").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "delta_lake").mkdir(parents=True, exist_ok=True)

        self._alert_log: List[Alert] = []
        self._delta_buffer: List[dict] = []
        logger.info("AlertManager initialized | cooldown=%.0fs", cooldown_sec)

    def _cooldown_key(self, service: str, metric: str) -> str:
        return f"{service}::{metric}"

    def _in_cooldown(self, service: str, metric: str, now: float) -> bool:
        key = self._cooldown_key(service, metric)
        last = self._last_alert.get(key, 0.0)
        return (now - last) < self.cooldown_sec

    def _make_alert(self, result: EnsembleResult, channel: str) -> Alert:
        now_str = datetime.utcfromtimestamp(result.timestamp / 1000).strftime("%Y-%m-%dT%H:%M:%SZ")
        msg = (
            f"[{result.severity.upper()}] Anomaly detected on {result.service} | "
            f"metric={result.metric_name} | value={result.value:.4f} | "
            f"confidence={result.ensemble_confidence:.2%} | "
            f"detectors={[k for k,v in result.votes.items() if v]}"
        )
        return Alert(
            alert_id=f"ALT-{result.event_id[:8]}",
            triggered_at=now_str,
            severity=result.severity,
            service=result.service,
            metric_name=result.metric_name,
            value=result.value,
            confidence=result.ensemble_confidence,
            channel=channel,
            message=msg,
            event_id=result.event_id
        )

    def _send_pagerduty(self, alert: Alert) -> None:
        """Simulate PagerDuty webhook call. In production: POST to PD Events API v2."""
        pd_payload = {
            "routing_key": "SIMULATED_PD_ROUTING_KEY",
            "event_action": "trigger",
            "dedup_key":    alert.alert_id,
            "payload": {
                "summary":   alert.message,
                "severity":  alert.severity.lower(),
                "source":    alert.service,
                "timestamp": alert.triggered_at,
                "custom_details": {
                    "metric":     alert.metric_name,
                    "value":      alert.value,
                    "confidence": alert.confidence
                }
            }
        }
        out = self.output_dir / "alerts" / "pagerduty" / f"{alert.alert_id}.json"
        with open(out, "w") as f:
            json.dump(pd_payload, f, indent=2)
        logger.warning("🚨 PAGERDUTY [%s] %s", alert.severity, alert.message)

    def _send_slack(self, alert: Alert) -> None:
        """Simulate Slack webhook. In production: POST to Slack Incoming Webhook URL."""
        emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡"}.get(alert.severity, "⚪")
        slack_payload = {
            "text": f"{emoji} *Anomaly Alert — {alert.severity}*",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn",
                 "text": f"{emoji} *{alert.severity} Anomaly* on `{alert.service}`\n"
                         f"*Metric:* `{alert.metric_name}` | *Value:* `{alert.value:.4f}`\n"
                         f"*Confidence:* `{alert.confidence:.2%}`\n"
                         f"*Time:* `{alert.triggered_at}`"}},
                {"type": "section", "text": {"type": "mrkdwn",
                 "text": f"*Message:* {alert.message}"}}
            ]
        }
        out = self.output_dir / "alerts" / "slack" / f"{alert.alert_id}.json"
        with open(out, "w") as f:
            json.dump(slack_payload, f, indent=2)
        logger.info("📢 SLACK [%s] %s", alert.severity, alert.message)

    def _write_delta(self, result: EnsembleResult) -> None:
        """Write anomalous event to Delta Lake audit table (partitioned by date/service)."""
        date_str = datetime.utcfromtimestamp(result.timestamp / 1000).strftime("%Y-%m-%d")
        partition = self.output_dir / "delta_lake" / f"date={date_str}" / f"service={result.service}"
        partition.mkdir(parents=True, exist_ok=True)

        record = {
            **result.to_dict(),
            "_delta_ts": time.time(),
            "_partition_date": date_str,
            "_partition_service": result.service
        }
        self._delta_buffer.append(record)

        # Write immediately (in production: batch write via Spark Delta)
        out = partition / f"{result.event_id}.json"
        with open(out, "w") as f:
            json.dump(record, f, indent=2, default=lambda x: bool(x) if isinstance(x, (bool,)) else str(x))

    def process(self, results: List[EnsembleResult]) -> List[Alert]:
        """Route detected anomalies to appropriate channels."""
        alerts_fired = []
        now = time.time()

        for result in results:
            if not result.is_anomaly:
                continue

            if self._in_cooldown(result.service, result.metric_name, now):
                continue  # suppress duplicate alerts

            # Route by severity
            if result.severity in self.pd_severities:
                channel = "pagerduty"
            elif result.severity in self.sl_severities:
                channel = "slack"
            else:
                channel = "log"
                logger.debug("LOW anomaly on %s/%s (conf=%.2f)",
                             result.service, result.metric_name, result.ensemble_confidence)

            alert = self._make_alert(result, channel)

            if channel == "pagerduty":
                self._send_pagerduty(alert)
            elif channel == "slack":
                self._send_slack(alert)

            # Always write to Delta Lake
            self._write_delta(result)

            # Update cooldown
            self._last_alert[self._cooldown_key(result.service, result.metric_name)] = now
            self._alert_log.append(alert)
            alerts_fired.append(alert)

        return alerts_fired

    def save_alert_log(self) -> str:
        """Persist all alerts to a summary file."""
        out_path = self.output_dir / "reports" / "alert_log.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([a.to_dict() for a in self._alert_log], f, indent=2)
        logger.info("Alert log saved: %d alerts → %s", len(self._alert_log), out_path)
        return str(out_path)

    def get_summary(self) -> dict:
        from collections import Counter
        return {
            "total_alerts": len(self._alert_log),
            "by_severity": dict(Counter(a.severity for a in self._alert_log)),
            "by_channel":  dict(Counter(a.channel  for a in self._alert_log)),
            "by_service":  dict(Counter(a.service  for a in self._alert_log)),
            "delta_records": len(self._delta_buffer)
        }
