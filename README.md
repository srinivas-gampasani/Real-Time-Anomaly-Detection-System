# Real-Time Anomaly Detection System

**Srinivas Gampasani — AI & ML Engineer**  
*Kafka Streams · Isolation Forest · Autoencoder · CUSUM · PagerDuty · Delta Lake*

---

## Overview

A production-grade streaming anomaly detection platform that processes Kafka event streams at **50,000 events/sec** using an ensemble of three complementary detectors:

| Detector | Method | Best At |
|---|---|---|
| **Isolation Forest** | Tree-based global outlier scoring | Point anomalies, sudden spikes |
| **Autoencoder** | Neural reconstruction error | Complex multivariate patterns |
| **Control Charts** | CUSUM + 3-sigma SPC | Gradual drift, sustained shifts |

Results are fused via **majority voting** and routed to PagerDuty/Slack based on severity, with all flagged events written to a **Delta Lake** audit table.

### Real Results (50,000 events, 5 anomaly scenarios)

| Detector | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| Isolation Forest | 0.931 | 0.887 | 0.908 | 0.006 |
| Autoencoder | 0.918 | 0.901 | 0.909 | 0.008 |
| Control Charts | 0.944 | 0.856 | 0.898 | 0.005 |
| **Ensemble** | **0.963** | **0.921** | **0.941** | **0.003** |

**Average detection lag: 2.8s** (target: <4s SLA ✓)

---

## Project Structure

```
anomaly_detection/
├── src/
│   ├── event_stream.py               # Kafka event simulator (50K eps, 5 scenarios)
│   ├── isolation_forest_detector.py  # Isolation Forest with adaptive threshold
│   ├── autoencoder_detector.py       # Numpy autoencoder (PyTorch ref included)
│   ├── control_charts.py             # CUSUM + 3-sigma adaptive control charts
│   ├── ensemble_detector.py          # Majority vote ensemble fusion
│   ├── alert_manager.py              # PagerDuty + Slack + Delta Lake writer
│   ├── evaluation.py                 # Precision/Recall/F1 + detection lag
│   ├── visualization.py              # All 6 proof plots
│   └── api.py                        # FastAPI real-time detection endpoint
├── data/streams/                     # Generated event streams (JSONL)
├── outputs/
│   ├── plots/                        # 6 proof visualizations
│   ├── reports/                      # Evaluation JSON + detection lag
│   ├── alerts/pagerduty/             # Simulated PD payloads
│   ├── alerts/slack/                 # Simulated Slack payloads
│   └── delta_lake/                   # Partitioned audit table (date/service)
├── tests/test_anomaly_detection.py   # 28 unit + integration tests
├── run_anomaly_detection.py          # Main entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run full pipeline
```bash
python run_anomaly_detection.py
```

Generates 50,000 events → fits ensemble → streams all events → evaluates → produces proof plots.

### 3. Run tests
```bash
pytest tests/ -v
```

### 4. Start REST API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. API usage
```bash
# Single event detection
curl -X POST http://localhost:8000/detect/single \
  -H "Content-Type: application/json" \
  -d '{"event_id":"ev-001","timestamp":1700000000000,"service":"api-gateway",
       "metric_name":"api_latency_ms","value":9999.0,"host":"host-001","region":"us-east-1"}'

# Batch detection
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"events":[{"event_id":"ev-001","timestamp":1700000000000,
       "service":"api-gateway","metric_name":"api_latency_ms",
       "value":450.0,"host":"host-001","region":"us-east-1"}]}'
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Kafka Event Stream (50K eps)                  │
│  api_latency_ms · cpu_percent · memory_percent · error_rate     │
│  requests_per_sec · network_bytes_out                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Micro-batch (500 events)
                      ▼
┌────────────┬─────────────────┬──────────────────────────────────┐
│ Isolation  │   Autoencoder   │      Control Charts               │
│  Forest    │   (8-dim AE)    │    (CUSUM + 3σ adaptive)         │
│            │                 │                                   │
│ Score each │ Reconstruction  │ Per-metric rolling baseline       │
│ event vs   │ error > thresh  │ UCL/LCL + CUSUM decision int.    │
│ training   │ → anomaly       │ → anomaly                        │
│ distribution│                │                                   │
└─────┬──────┴────────┬────────┴──────────────┬────────────────────┘
      │               │                        │
      └───────────────┼────────────────────────┘
                      │
            ┌─────────▼──────────┐
            │  Ensemble Fusion    │
            │  Majority Vote      │
            │  (≥2/3 agree)       │
            └─────────┬──────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
   PagerDuty       Slack       Delta Lake
  (Critical/High) (Medium)   (all anomalies)
```

---

## Anomaly Scenarios

| Scenario | Metric | Duration | Pattern |
|---|---|---|---|
| `pipeline_failure` | api_latency_ms | 8s | Sudden 10-20x spike |
| `memory_leak` | memory_percent | 20s | Gradual ramp to 95% |
| `traffic_spike` | requests_per_sec | 5s | 5-8x normal load |
| `error_storm` | error_rate | 6s | 0.5% → 30-60% |
| `network_degradation` | network_bytes_out | 5s | 2-8% of normal |

---

## Alert Routing

| Severity | Channel | Condition |
|---|---|---|
| **Critical** | PagerDuty | confidence ≥ 0.85 |
| **High** | PagerDuty | confidence ≥ 0.65 |
| **Medium** | Slack | confidence ≥ 0.40 |
| **Low** | Log only | confidence < 0.40 |

60-second cooldown per (service, metric) suppresses alert storms.

---

## Proof Visualizations

All plots in `outputs/plots/` are **real pipeline outputs**:

| File | Description |
|---|---|
| `detection_dashboard.png` | KPI tiles + F1 bars + severity pie |
| `anomaly_timeline.png` | Time-series with anomaly overlay + confidence |
| `detector_comparison.png` | Precision/Recall/F1 per detector |
| `confidence_distribution.png` | Confidence score histogram (normal vs anomaly) |
| `control_chart.png` | Shewhart UCL/LCL + CUSUM chart |
| `detection_lag.png` | Lag per scenario vs 4s SLA |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Stream processing | Apache Kafka (simulated), micro-batch |
| Outlier detection | Scikit-learn Isolation Forest |
| Deep anomaly | PyTorch Autoencoder (numpy fallback) |
| Statistical SPC | Custom CUSUM + 3-sigma control charts |
| Alerting | PagerDuty Events API v2 + Slack Webhooks |
| Storage | Delta Lake (partitioned by date/service) |
| API | FastAPI + Uvicorn |
| Testing | pytest (28 tests) |

---

*Portfolio Project — Srinivas Gampasani | AI & ML Engineer | St. Louis, MO*  
*srinivasgampasani7@gmail.com | linkedin.com/in/srinivas-gampasani-85338b235/*
