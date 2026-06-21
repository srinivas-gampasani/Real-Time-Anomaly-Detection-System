"""
run_anomaly_detection.py
=========================
Full pipeline execution:
  1. Generate event stream (50K events with 5 anomaly scenarios)
  2. Fit ensemble on baseline data
  3. Stream events through detector in micro-batches
  4. Collect alerts (PagerDuty + Slack simulation)
  5. Evaluate all detectors (Precision/Recall/F1)
  6. Compute detection lag per scenario
  7. Generate all proof visualizations

Usage:
    python run_anomaly_detection.py
"""

import sys
import json
import numpy as np

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (bool,)): return bool(obj)
        if hasattr(obj, 'item'): return obj.item()
        if hasattr(obj, 'tolist'): return obj.tolist()
        return super().default(obj)
import logging
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
Path("outputs/reports").mkdir(parents=True, exist_ok=True)
Path("outputs/plots").mkdir(parents=True, exist_ok=True)
Path("outputs/alerts").mkdir(parents=True, exist_ok=True)
Path("outputs/delta_lake").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/pipeline.log", mode="w")
    ]
)
logger = logging.getLogger("run_anomaly_detection")


def main():
    print("\n" + "="*70)
    print("   REAL-TIME ANOMALY DETECTION SYSTEM")
    print("   Srinivas Gampasani — AI & ML Engineer")
    print("="*70)

    # ── 1. Generate event stream ─────────────────────────────────────────────
    print("\n[1/7] Generating event stream (50K events, 5 anomaly scenarios)...")
    from src.event_stream import EventStreamSimulator, BASELINES

    sim = EventStreamSimulator(events_per_second=50000)
    sim.add_scenario("pipeline_failure",    5.0,   8.0,  "api_latency_ms",    "api-gateway")
    sim.add_scenario("memory_leak",         15.0,  20.0, "memory_percent",     "order-service")
    sim.add_scenario("traffic_spike",       40.0,  5.0,  "requests_per_sec",   "payment-service")
    sim.add_scenario("error_storm",         55.0,  6.0,  "error_rate",         "auth-service")
    sim.add_scenario("network_degradation", 70.0,  5.0,  "network_bytes_out",  "inventory-service")

    total_events = 50000
    simulation_duration_sec = 80.0   # covers all 5 scenarios (last ends at 75s)
    all_events = []
    batch_size = 1000
    n_batches  = total_events // batch_size
    t_start_ms = time.time() * 1000

    for b in range(n_batches):
        # Map batch index to simulated elapsed time (0..80s)
        elapsed = b / n_batches * simulation_duration_sec
        batch = sim.generate_batch(
            batch_size=batch_size,
            start_timestamp_ms=t_start_ms + elapsed * 1000,
            elapsed_sec=elapsed
        )
        all_events.extend([e.to_dict() for e in batch])

    n_anom = sum(1 for e in all_events if e["is_anomaly"])
    print(f"      ✓ Total events: {len(all_events):,}")
    print(f"      ✓ Injected anomalies: {n_anom:,} ({n_anom/len(all_events):.2%})")

    # Save stream
    stream_path = "data/streams/event_stream.jsonl"
    Path(stream_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stream_path, "w") as f:
        for ev in all_events:
            f.write(json.dumps(ev) + "\n")

    # ── 2. Baseline split ────────────────────────────────────────────────────
    print("\n[2/7] Splitting baseline / stream data...")
    # First 3000 clean events as baseline (before first anomaly at t=5s)
    baseline = [e for e in all_events[:3000] if not e["is_anomaly"]]
    stream   = all_events[3000:]
    print(f"      ✓ Baseline: {len(baseline):,} clean events")
    print(f"      ✓ Stream:   {len(stream):,} events")

    # ── 3. Fit ensemble ──────────────────────────────────────────────────────
    print("\n[3/7] Fitting ensemble detector (IF + Autoencoder + ControlCharts)...")
    from src.ensemble_detector import EnsembleDetector, FusionStrategy

    ensemble = EnsembleDetector(
        fusion_strategy=FusionStrategy.MAJORITY_VOTE,
        if_contamination=0.02,
        ae_latent_dim=8,
        cc_sigma=3.0,
        cc_cusum_h=5.0
    )
    t0 = time.time()
    ensemble.fit(baseline, ae_epochs=80)
    fit_time = time.time() - t0
    print(f"      ✓ Ensemble fitted in {fit_time:.2f}s")

    # ── 4. Stream processing ─────────────────────────────────────────────────
    print("\n[4/7] Processing event stream in micro-batches...")
    from src.alert_manager import AlertManager

    alert_mgr = AlertManager(output_dir="outputs", cooldown_sec=30.0)

    all_results = []
    total_alerts = 0
    micro_batch  = 500
    t_proc_start = time.time()

    for i in range(0, len(stream), micro_batch):
        batch  = stream[i:i + micro_batch]
        results = ensemble.score_batch(batch)
        alerts  = alert_mgr.process(results)
        all_results.extend(results)
        total_alerts += len(alerts)

        if i % 10000 == 0:
            done = i + len(batch)
            elapsed = time.time() - t_proc_start
            eps = done / elapsed if elapsed > 0 else 0
            print(f"      Processed {done:>6,}/{len(stream):,}  "
                  f"Alerts: {total_alerts}  "
                  f"Throughput: {eps:,.0f} events/sec")

    proc_time = time.time() - t_proc_start
    actual_eps = len(stream) / proc_time
    print(f"\n      ✓ Processed {len(stream):,} events in {proc_time:.2f}s")
    print(f"      ✓ CPU throughput: {actual_eps:,.0f} events/sec")
    print(f"      ✓ Total alerts fired: {total_alerts}")
    alert_mgr.save_alert_log()

    # ── 5. Evaluate all detectors ────────────────────────────────────────────
    print("\n[5/7] Evaluating detectors vs ground truth...")
    from src.evaluation import compute_metrics, print_eval_table
    from src.isolation_forest_detector import IsolationForestDetector
    from src.autoencoder_detector import AutoencoderDetector
    from src.control_charts import ControlChartsDetector
    import numpy as np

    ground_truth = [e["is_anomaly"] for e in stream]

    # Isolation Forest alone
    if_det = IsolationForestDetector(contamination=0.02, n_estimators=200)
    if_det.fit(np.array([e["value"] for e in baseline]))
    if_results  = if_det.score_batch(stream)
    if_preds    = [r.is_anomaly for r in if_results]
    if_confs    = [r.confidence  for r in if_results]

    # Autoencoder alone
    ae_det = AutoencoderDetector(latent_dim=8)
    ae_det.fit(baseline, epochs=80)
    ae_results = ae_det.score_batch(stream)
    ae_preds   = [r.is_anomaly for r in ae_results]
    ae_confs   = [r.confidence  for r in ae_results]

    # Control Charts alone
    cc_det = ControlChartsDetector()
    cc_det.fit(baseline)
    cc_results = cc_det.score_batch(stream)
    cc_preds   = [r.is_anomaly for r in cc_results]
    cc_confs   = [r.confidence  for r in cc_results]

    # Ensemble
    ens_preds = [r.is_anomaly        for r in all_results]
    ens_confs = [r.ensemble_confidence for r in all_results]
    anom_types = [e.get("anomaly_type") for e in stream]

    eval_results = {
        "IsolationForest": compute_metrics("IsolationForest", if_preds, ground_truth, if_confs, anom_types),
        "Autoencoder":     compute_metrics("Autoencoder",     ae_preds, ground_truth, ae_confs, anom_types),
        "ControlCharts":   compute_metrics("ControlCharts",   cc_preds, ground_truth, cc_confs, anom_types),
        "Ensemble":        compute_metrics("Ensemble",        ens_preds, ground_truth, ens_confs, anom_types),
    }
    print_eval_table(eval_results)

    # Save eval report
    with open("outputs/reports/evaluation_report.json", "w") as f:
        json.dump({k: v.to_dict() for k, v in eval_results.items()}, f, indent=2, cls=_SafeEncoder)

    # ── 6. Detection lag ─────────────────────────────────────────────────────
    print("\n[6/7] Computing detection lag per scenario...")
    # Compute start timestamps for each anomaly type
    anomaly_starts = {}
    for ev in stream:
        at = ev.get("anomaly_type")
        if at and at not in anomaly_starts:
            anomaly_starts[at] = ev["timestamp"]

    first_detections = {}
    for ev, pred in zip(stream, ens_preds):
        at = ev.get("anomaly_type")
        if at and pred and at not in first_detections:
            first_detections[at] = ev["timestamp"]

    lag_data = {}
    for atype, start_ms in anomaly_starts.items():
        if atype in first_detections:
            lag_data[atype] = round((first_detections[atype] - start_ms) / 1000, 2)
        else:
            lag_data[atype] = None

    print("\n  Detection Lag per Scenario:")
    avg_lags = []
    for sc, lag in lag_data.items():
        status = f"{lag:.2f}s" if lag is not None else "NOT DETECTED"
        flag   = "✓" if lag is not None and lag <= 4.0 else "⚠"
        print(f"    {flag} {sc:<28} {status}")
        if lag is not None:
            avg_lags.append(lag)

    avg_lag = sum(avg_lags) / len(avg_lags) if avg_lags else 0.0
    print(f"\n    Average detection lag: {avg_lag:.2f}s (SLA: 4s)")

    with open("outputs/reports/detection_lag.json", "w") as f:
        json.dump({k: (float(v) if v is not None else None) for k,v in lag_data.items()}, f, indent=2)

    # ── 7. Visualizations ────────────────────────────────────────────────────
    print("\n[7/7] Generating proof visualizations...")
    from src.visualization import (
        plot_anomaly_timeline,
        plot_detector_comparison,
        plot_confidence_distribution,
        plot_control_chart,
        plot_detection_lag,
        plot_dashboard
    )

    all_results_dicts = [r.to_dict() for r in all_results]
    alert_summary = alert_mgr.get_summary()

    metrics_summary = {
        "total_events":     len(all_events),
        "total_anomalies":  sum(r.is_anomaly for r in all_results),
        "avg_lag_sec":      avg_lag,
        "throughput_eps":   round(actual_eps),
        "fit_time_sec":     round(fit_time, 2),
    }

    p1 = plot_anomaly_timeline(stream, all_results_dicts, "api_latency_ms")
    p2 = plot_detector_comparison(eval_results)
    p3 = plot_confidence_distribution(all_results_dicts)
    p4 = plot_control_chart(stream, "memory_percent")
    p5 = plot_detection_lag(lag_data)
    p6 = plot_dashboard(metrics_summary, eval_results, alert_summary)

    print(f"\n  ✓ Plots saved:")
    for p in [p1, p2, p3, p4, p5, p6]:
        if p:
            print(f"    → {p}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    ens = eval_results["Ensemble"]
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE — RESULTS SUMMARY")
    print("="*70)
    print(f"  Events processed    : {len(all_events):,}")
    print(f"  Throughput          : {actual_eps:,.0f} events/sec (CPU)")
    print(f"  Anomalies detected  : {sum(r.is_anomaly for r in all_results):,}")
    print(f"  Alerts fired        : {total_alerts}")
    print(f"\n  Ensemble Performance:")
    print(f"    Precision         : {ens.precision:.4f}")
    print(f"    Recall            : {ens.recall:.4f}")
    print(f"    F1 Score          : {ens.f1:.4f}")
    print(f"    FPR               : {ens.false_positive_rate:.4f}")
    print(f"\n  Detection Lag       : {avg_lag:.2f}s avg (target: <4s)")
    print(f"  Alert Channels      : {alert_summary.get('by_channel', {})}")
    print(f"  Delta Lake records  : {alert_summary.get('delta_records', 0)}")
    print(f"\n  Outputs in: outputs/")
    print(f"    ├── plots/               (6 proof visualizations)")
    print(f"    ├── reports/             (eval + lag JSON)")
    print(f"    ├── alerts/pagerduty/    (PagerDuty payloads)")
    print(f"    ├── alerts/slack/        (Slack payloads)")
    print(f"    └── delta_lake/          (partitioned audit table)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
