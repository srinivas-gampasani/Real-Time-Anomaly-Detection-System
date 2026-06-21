"""
Event Stream Simulator
========================
Simulates Kafka event streams at 50K events/sec for:
  - API latency metrics
  - CPU / Memory usage
  - Error rates
  - Transaction throughput
  - Network I/O

Injects realistic anomaly scenarios:
  1. Data pipeline failure (spike + sustained high latency)
  2. Memory leak (gradual ramp-up)
  3. Traffic spike (sudden throughput surge)
  4. Error storm (error rate explosion)
  5. Network degradation (packet loss / jitter)
"""

import json
import time
import random
import hashlib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Generator
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)


@dataclass
class MetricEvent:
    event_id: str
    timestamp: float           # Unix epoch ms
    service: str
    metric_name: str
    value: float
    unit: str
    host: str
    region: str
    is_anomaly: bool = False   # ground truth label
    anomaly_type: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict())


SERVICES   = ["api-gateway", "order-service", "payment-service", "inventory-service", "auth-service"]
HOSTS      = [f"host-{i:03d}" for i in range(1, 11)]
REGIONS    = ["us-east-1", "us-west-2", "eu-west-1"]

# Normal operating baselines per metric
BASELINES = {
    "api_latency_ms":     {"mean": 45.0,  "std": 8.0,   "unit": "ms"},
    "cpu_percent":        {"mean": 35.0,  "std": 6.0,   "unit": "%"},
    "memory_percent":     {"mean": 55.0,  "std": 5.0,   "unit": "%"},
    "error_rate":         {"mean": 0.005, "std": 0.002, "unit": "ratio"},
    "requests_per_sec":   {"mean": 1200,  "std": 120,   "unit": "rps"},
    "network_bytes_out":  {"mean": 5e6,   "std": 8e5,   "unit": "bytes/s"},
}


def _make_event_id(timestamp: float, service: str, metric: str) -> str:
    raw = f"{timestamp}-{service}-{metric}-{random.random()}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def generate_normal_event(
    timestamp: float,
    service: str,
    metric_name: str,
    host: str,
    region: str
) -> MetricEvent:
    bl = BASELINES[metric_name]
    value = np.random.normal(bl["mean"], bl["std"])
    value = max(0.0, value)
    return MetricEvent(
        event_id=_make_event_id(timestamp, service, metric_name),
        timestamp=timestamp,
        service=service,
        metric_name=metric_name,
        value=round(value, 4),
        unit=bl["unit"],
        host=host,
        region=region,
        is_anomaly=False
    )


# ─── Anomaly Scenario Generators ──────────────────────────────────────────────

def inject_pipeline_failure(base_event: MetricEvent, severity: float = 1.0) -> MetricEvent:
    """Sudden spike: latency 10-20x normal."""
    base_event.value = round(
        BASELINES["api_latency_ms"]["mean"] * (10 + severity * 10) + np.random.normal(0, 20), 2
    )
    base_event.is_anomaly = True
    base_event.anomaly_type = "pipeline_failure"
    return base_event


def inject_memory_leak(base_event: MetricEvent, progress: float = 0.5) -> MetricEvent:
    """Gradual memory ramp-up. progress ∈ [0, 1]."""
    base_event.value = round(
        BASELINES["memory_percent"]["mean"] + progress * 40 + np.random.normal(0, 1), 2
    )
    base_event.is_anomaly = base_event.value > 85
    if base_event.is_anomaly:
        base_event.anomaly_type = "memory_leak"
    return base_event


def inject_traffic_spike(base_event: MetricEvent) -> MetricEvent:
    """Requests spike 5-8x normal."""
    base_event.value = round(
        BASELINES["requests_per_sec"]["mean"] * np.random.uniform(5, 8), 1
    )
    base_event.is_anomaly = True
    base_event.anomaly_type = "traffic_spike"
    return base_event


def inject_error_storm(base_event: MetricEvent) -> MetricEvent:
    """Error rate jumps from 0.5% to 30-60%."""
    base_event.value = round(np.random.uniform(0.30, 0.60), 4)
    base_event.is_anomaly = True
    base_event.anomaly_type = "error_storm"
    return base_event


def inject_network_degradation(base_event: MetricEvent) -> MetricEvent:
    """Network bytes drop to near-zero (packet loss)."""
    base_event.value = round(
        BASELINES["network_bytes_out"]["mean"] * np.random.uniform(0.02, 0.08), 1
    )
    base_event.is_anomaly = True
    base_event.anomaly_type = "network_degradation"
    return base_event


# ─── Stream Generator ─────────────────────────────────────────────────────────

class EventStreamSimulator:
    """
    Simulates a Kafka event stream with configurable anomaly injection.
    Designed to mimic 50K events/sec production throughput.
    """

    def __init__(self, events_per_second: int = 50000, scenario_seed: int = 42):
        self.eps = events_per_second
        self.rng = np.random.RandomState(scenario_seed)
        self._anomaly_scenarios = []

    def add_scenario(self, name: str, start_offset_sec: float,
                     duration_sec: float, metric: str, service: str = None):
        self._anomaly_scenarios.append({
            "name": name,
            "start": start_offset_sec,
            "end": start_offset_sec + duration_sec,
            "metric": metric,
            "service": service or random.choice(SERVICES)
        })

    def _active_scenario(self, elapsed_sec: float, service: str, metric: str):
        for sc in self._anomaly_scenarios:
            if (sc["start"] <= elapsed_sec <= sc["end"]
                    and sc["metric"] == metric
                    and (sc["service"] is None or sc["service"] == service)):
                progress = (elapsed_sec - sc["start"]) / (sc["end"] - sc["start"])
                return sc["name"], progress
        return None, 0.0

    def generate_batch(
        self,
        batch_size: int = 1000,
        start_timestamp_ms: float = None,
        elapsed_sec: float = 0.0
    ) -> List[MetricEvent]:
        """Generate a batch of events (simulates one Kafka micro-batch)."""
        if start_timestamp_ms is None:
            start_timestamp_ms = time.time() * 1000

        events = []
        for i in range(batch_size):
            ts = start_timestamp_ms + (i / self.eps) * 1000
            service = random.choice(SERVICES)
            metric  = random.choice(list(BASELINES.keys()))
            host    = random.choice(HOSTS)
            region  = random.choice(REGIONS)

            event = generate_normal_event(ts, service, metric, host, region)

            scenario_name, progress = self._active_scenario(elapsed_sec, service, metric)
            if scenario_name == "pipeline_failure":
                event = inject_pipeline_failure(event, severity=progress)
            elif scenario_name == "memory_leak":
                event = inject_memory_leak(event, progress=progress)
            elif scenario_name == "traffic_spike":
                event = inject_traffic_spike(event)
            elif scenario_name == "error_storm":
                event = inject_error_storm(event)
            elif scenario_name == "network_degradation":
                event = inject_network_degradation(event)

            events.append(event)
        return events

    def stream(
        self,
        total_seconds: float = 60.0,
        batch_size: int = 1000
    ) -> Generator[List[MetricEvent], None, None]:
        """
        Generator that yields batches of events over a simulated time window.
        Each yield = one Kafka micro-batch.
        """
        start_ts = time.time() * 1000
        batches = int(total_seconds * self.eps / batch_size)
        for b in range(batches):
            elapsed = b * batch_size / self.eps
            batch = self.generate_batch(
                batch_size=batch_size,
                start_timestamp_ms=start_ts + elapsed * 1000,
                elapsed_sec=elapsed
            )
            yield batch

    def save_dataset(
        self,
        output_path: str,
        total_events: int = 50000,
        include_scenarios: bool = True
    ) -> dict:
        """
        Save a labeled dataset for model training and evaluation.
        Returns summary statistics.
        """
        if include_scenarios:
            self.add_scenario("pipeline_failure", 5.0,  8.0,  "api_latency_ms",    "api-gateway")
            self.add_scenario("memory_leak",      15.0, 20.0, "memory_percent",     "order-service")
            self.add_scenario("traffic_spike",    40.0, 5.0,  "requests_per_sec",   "payment-service")
            self.add_scenario("error_storm",      55.0, 6.0,  "error_rate",         "auth-service")
            self.add_scenario("network_degradation", 70.0, 5.0, "network_bytes_out", "inventory-service")

        all_events = []
        batch_size = 1000
        batches = total_events // batch_size
        start_ts = time.time() * 1000

        for b in range(batches):
            elapsed = b * batch_size / self.eps
            batch = self.generate_batch(
                batch_size=batch_size,
                start_timestamp_ms=start_ts + elapsed * 1000,
                elapsed_sec=elapsed
            )
            all_events.extend(batch)

        # Save as JSON lines
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ev in all_events:
                f.write(ev.to_json() + "\n")

        n_anomalies = sum(1 for e in all_events if e.is_anomaly)
        summary = {
            "total_events": len(all_events),
            "anomalies": n_anomalies,
            "normal": len(all_events) - n_anomalies,
            "anomaly_rate": round(n_anomalies / len(all_events), 4),
            "anomaly_types": list({e.anomaly_type for e in all_events if e.is_anomaly}),
            "metrics": list(BASELINES.keys()),
            "services": SERVICES,
        }
        print(f"Dataset saved: {output_path}")
        print(f"  Total: {summary['total_events']:,}  Anomalies: {n_anomalies:,}  Rate: {summary['anomaly_rate']:.2%}")
        return summary


if __name__ == "__main__":
    sim = EventStreamSimulator(events_per_second=50000)
    sim.save_dataset("data/streams/event_stream.jsonl", total_events=50000)
