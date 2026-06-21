"""
FastAPI Real-Time Anomaly Detection Service
============================================
Endpoints:
  POST /detect          — Score a batch of events
  POST /detect/single   — Score a single event
  GET  /health          — Health check
  GET  /stats           — System statistics
  GET  /alerts          — Recent alerts

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import time
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Anomaly Detection API",
    description="Ensemble Isolation Forest + Autoencoder + Control Charts",
    version="1.0.0"
)

_detector  = None
_alert_mgr = None
_stats     = {"requests": 0, "anomalies_detected": 0, "alerts_fired": 0, "uptime_start": time.time()}


class EventPayload(BaseModel):
    event_id: str
    timestamp: float
    service: str
    metric_name: str
    value: float
    host: str = "unknown"
    region: str = "us-east-1"


class DetectionResponse(BaseModel):
    event_id: str
    service: str
    metric_name: str
    value: float
    is_anomaly: bool
    severity: str
    ensemble_confidence: float
    votes: Dict[str, bool]
    latency_ms: float


class BatchRequest(BaseModel):
    events: List[EventPayload] = Field(..., min_length=1, max_length=10000)


class BatchResponse(BaseModel):
    total: int
    anomalies: int
    results: List[DetectionResponse]
    batch_latency_ms: float


@app.on_event("startup")
def startup():
    """Load pre-fitted detector at startup."""
    import json
    from pathlib import Path
    from src.ensemble_detector import EnsembleDetector, FusionStrategy
    from src.alert_manager import AlertManager
    from src.event_stream import EventStreamSimulator

    global _detector, _alert_mgr

    logger.info("Loading ensemble detector...")
    _detector = EnsembleDetector(fusion_strategy=FusionStrategy.MAJORITY_VOTE)
    _alert_mgr = AlertManager(output_dir="outputs")

    # Fit on synthetic baseline
    sim = EventStreamSimulator()
    baseline_batch = sim.generate_batch(batch_size=2000, elapsed_sec=0.0)
    baseline_events = [e.to_dict() for e in baseline_batch]
    _detector.fit(baseline_events, ae_epochs=30)
    logger.info("Detector ready.")
    _stats["uptime_start"] = time.time()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "detector_ready": _detector is not None and _detector._fitted,
        "uptime_sec": round(time.time() - _stats["uptime_start"], 1)
    }


@app.get("/stats")
def stats():
    return {
        **_stats,
        "alert_summary": _alert_mgr.get_summary() if _alert_mgr else {},
        "uptime_sec": round(time.time() - _stats["uptime_start"], 1)
    }


@app.get("/alerts")
def get_alerts(limit: int = 20):
    if not _alert_mgr:
        raise HTTPException(503, "Alert manager not ready")
    alerts = _alert_mgr._alert_log[-limit:]
    return {"count": len(alerts), "alerts": [a.to_dict() for a in alerts]}


@app.post("/detect/single", response_model=DetectionResponse)
def detect_single(event: EventPayload):
    if not _detector or not _detector._fitted:
        raise HTTPException(503, "Detector not ready")

    t0 = time.time()
    ev_dict = event.model_dump()
    results = _detector.score_batch([ev_dict])
    r = results[0]
    latency = (time.time() - t0) * 1000

    _stats["requests"] += 1
    if r.is_anomaly:
        _stats["anomalies_detected"] += 1
        alerts = _alert_mgr.process([r])
        _stats["alerts_fired"] += len(alerts)

    return DetectionResponse(
        event_id=r.event_id,
        service=r.service,
        metric_name=r.metric_name,
        value=r.value,
        is_anomaly=r.is_anomaly,
        severity=r.severity,
        ensemble_confidence=r.ensemble_confidence,
        votes=r.votes,
        latency_ms=round(latency, 2)
    )


@app.post("/detect", response_model=BatchResponse)
def detect_batch(req: BatchRequest):
    if not _detector or not _detector._fitted:
        raise HTTPException(503, "Detector not ready")

    t0 = time.time()
    ev_dicts = [e.model_dump() for e in req.events]
    results  = _detector.score_batch(ev_dicts)
    batch_latency = (time.time() - t0) * 1000

    anomalies = [r for r in results if r.is_anomaly]
    if anomalies:
        alerts = _alert_mgr.process(anomalies)
        _stats["alerts_fired"] += len(alerts)

    _stats["requests"] += 1
    _stats["anomalies_detected"] += len(anomalies)

    return BatchResponse(
        total=len(results),
        anomalies=len(anomalies),
        batch_latency_ms=round(batch_latency, 2),
        results=[
            DetectionResponse(
                event_id=r.event_id,
                service=r.service,
                metric_name=r.metric_name,
                value=r.value,
                is_anomaly=r.is_anomaly,
                severity=r.severity,
                ensemble_confidence=r.ensemble_confidence,
                votes=r.votes,
                latency_ms=round(batch_latency / len(results), 3)
            ) for r in results
        ]
    )
