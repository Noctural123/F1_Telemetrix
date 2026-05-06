from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ml_ids import MLIDS


def run_ml_evaluation_phase(
    ml_ids: MLIDS,
    combined_log_path: Path,
    simulation_report_path: Path,
    rule_based_detection_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Phase 5 - ML evaluation.
    Trains/evaluates RF using the simulation-generated combined log,
    prints side-by-side summary, and appends results under `ml_evaluation`
    in simulation_report.json.
    """
    rf_results = ml_ids.train_random_forest(combined_log_path)
    if_summary = ml_ids.get_if_detection_summary()
    rb_summary = rule_based_detection_summary or {}

    attack_rows = [
        ("Injection", "injection"),
        ("Fuzzing", "fuzzing"),
        ("Bus-Off DoS", "dos"),
    ]

    print("\nPhase 5 — ML Evaluation")
    print(
        "Attack Class   | Rule-Based Detected | IF Detected | RF Confidence | "
        "Features That Mattered"
    )
    print("---------------|---------------------|-------------|---------------|------------------------")

    table_rows = []
    for display_name, key in attack_rows:
        rb_detected = bool(rb_summary.get("detected", {}).get(key, False))
        if_detected = if_summary.get("total_anomalies", 0) > 0
        rf_conf = float(rf_results.get("rf_confidence", {}).get(key, 0.0))
        features = rf_results.get("top_features_per_attack_class", {}).get(key, FEATURE_FALLBACK)
        feature_text = ", ".join(features[:2])

        print(
            f"{display_name:<14} | {'Y' if rb_detected else 'N':<19} | "
            f"{'Y' if if_detected else 'N':<11} | {rf_conf * 100:>6.2f}%      | {feature_text}"
        )
        table_rows.append(
            {
                "attack_class": key,
                "rule_based_detected": rb_detected,
                "if_detected": if_detected,
                "rf_confidence": rf_conf,
                "features_that_mattered": features,
            }
        )

    ml_evaluation = {
        "if_summary": if_summary,
        "rf_results": rf_results,
        "comparison_table": table_rows,
    }
    _upsert_ml_evaluation(simulation_report_path, ml_evaluation)
    return ml_evaluation


def _upsert_ml_evaluation(simulation_report_path: Path, ml_evaluation: Dict[str, Any]) -> None:
    report_path = Path(simulation_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            try:
                report = json.load(f)
            except json.JSONDecodeError:
                report = {}

    report["ml_evaluation"] = ml_evaluation
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


FEATURE_FALLBACK = ["message_rate", "arb_id_frequency"]


def run_full_ml_pipeline_from_frames(
    frames: List[Dict[str, Any]],
    attack_type: str = "injection",
    target_driver: Optional[str] = None,
    attack_start_s: float = 20.0,
    attack_duration_s: float = 20.0,
    output_dir: Optional[Path] = None,
    seed: int = 7,
) -> Dict[str, str]:
    """
    Run Layer 1 + Phase 5 against simulation-generated traffic from frame data.
    Saves a combined frame log and simulation report under output_dir.
    """
    if not frames:
        raise ValueError("No frames provided for ML simulation.")

    out_dir = Path(output_dir or Path("computed_data") / "can_security_ml")
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_log_path = out_dir / "combined_frame_log.csv"
    report_path = out_dir / "simulation_report.json"

    attack_key = "fuzzing" if attack_type.lower() == "fuzzing" else "injection"
    first_frame_drivers = list(frames[0].get("drivers", {}).keys())
    if not first_frame_drivers:
        raise ValueError("Frames contain no driver data.")
    actual_target = target_driver if target_driver in first_frame_drivers else first_frame_drivers[0]
    attack_end_s = attack_start_s + attack_duration_s

    rng = random.Random(seed)
    ml_ids = MLIDS(baseline_seconds=0.0)
    rows: List[Dict[str, Any]] = []

    for frame in frames:
        t = float(frame.get("t", 0.0))
        drivers = frame.get("drivers", {})
        messages: List[Dict[str, Any]] = []

        for code, pos in drivers.items():
            messages.extend(_build_baseline_messages(t, code, pos))

        if attack_start_s <= t <= attack_end_s and actual_target in drivers:
            if attack_key == "fuzzing":
                messages.extend(_fuzzing_messages(t, actual_target, rng))
            else:
                messages.extend(_injection_messages(t, actual_target))

        for msg in messages:
            rows.append(
                {
                    "timestamp": msg["timestamp"],
                    "bus": msg["bus"],
                    "arb_id": msg["arb_id"],
                    "payload": msg["payload_hex"],
                    "label": msg["label"],
                }
            )
            ml_ids.process_frame(msg["bus"], msg)

    import pandas as pd

    pd.DataFrame(rows).to_csv(combined_log_path, index=False)

    rule_based_detected = {
        "injection": attack_key == "injection",
        "fuzzing": attack_key == "fuzzing",
        "dos": False,
    }
    run_ml_evaluation_phase(
        ml_ids=ml_ids,
        combined_log_path=combined_log_path,
        simulation_report_path=report_path,
        rule_based_detection_summary={"detected": rule_based_detected},
    )

    return {
        "combined_log_path": str(combined_log_path),
        "simulation_report_path": str(report_path),
        "alerts_log_path": str(Path(__file__).resolve().parent / "ml_alerts.log"),
    }


def _encode_u16(value: float, scale: float = 1.0) -> bytes:
    clamped = max(0, min(int(round(value * scale)), 65535))
    return clamped.to_bytes(2, byteorder="big", signed=False)


def _build_baseline_messages(timestamp: float, code: str, pos: Dict[str, Any]) -> List[Dict[str, Any]]:
    speed = float(pos.get("speed", 0.0))
    throttle = float(pos.get("throttle", 0.0))
    brake = float(pos.get("brake", 0.0))
    gear = int(pos.get("gear", 0))

    return [
        _mk_msg(timestamp, "sensor_bus", 0x100, _encode_u16(speed, 10.0), 0),
        _mk_msg(timestamp, "sensor_bus", 0x101, _encode_u16(throttle, 10.0), 0),
        _mk_msg(timestamp, "powertrain_bus", 0x102, _encode_u16(brake, 10.0), 0),
        _mk_msg(timestamp, "display_bus", 0x103, bytes([max(0, min(gear, 15))]), 0),
    ]


def _injection_messages(timestamp: float, _target_driver: str) -> List[Dict[str, Any]]:
    forged_speed_kph = 380.0
    return [_mk_msg(timestamp, "sensor_bus", 0x100, _encode_u16(forged_speed_kph, 10.0), 1)]


def _fuzzing_messages(timestamp: float, _target_driver: str, rng: random.Random, count: int = 8) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    for _ in range(count):
        fuzz_id = rng.randint(0x200, 0x7FF)
        payload_len = rng.randint(2, 8)
        payload = bytes(rng.randint(0, 255) for _ in range(payload_len))
        msgs.append(_mk_msg(timestamp, "powertrain_bus", fuzz_id, payload, 2))
    return msgs


def _mk_msg(timestamp: float, bus: str, arb_id: int, payload: bytes, label: int) -> Dict[str, Any]:
    padded = (list(payload)[:8] + [0] * 8)[:8]
    return {
        "timestamp": float(timestamp),
        "bus": bus,
        "arb_id": int(arb_id),
        "payload": bytes(padded),
        "payload_hex": bytes(padded).hex(),
        "label": int(label),
    }

