from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ml_ids import MLIDS
from .rule_based_ids import RuleBasedIDS


FEATURE_FALLBACK = ["message_rate", "arb_id_frequency"]
ATTACK_LABEL_BY_NAME = {"normal": 0, "injection": 1, "fuzzing": 2, "dos": 3}

# Per-driver and per-attack timestamp jitter (in seconds). Small enough to stay
# inside one telemetry frame interval (40 ms at FPS=25) but large enough to keep
# inter-arrival features from collapsing to 0. The effective per-driver jitter
# is computed adaptively so it remains valid for any driver count.
DRIVER_JITTER_S = 1e-3
ATTACK_JITTER_S = 1e-4
# Cap how far the (driver-jitter * n_drivers) span can stretch so attack/baseline
# messages never bleed into the next telemetry frame. ~32 ms at FPS=25 (40 ms
# frame interval) gives the frequency rule a clean 1-second window.
MAX_FRAME_JITTER_SPAN_S = 32e-3


def run_ml_evaluation_phase(
    ml_ids: MLIDS,
    combined_log_path: Path,
    simulation_report_path: Path,
    rule_based_summary: Optional[Dict[str, Any]] = None,
    attack_simulated: str = "injection",
) -> Dict[str, Any]:
    """
    Phase 5 - hybrid IDS evaluation.

    Trains/evaluates RF on the simulation log, joins per-class detection counts
    from the streaming Rule-Based IDS and Isolation Forest, prints a side-by-side
    summary, and appends results to `simulation_report.json` under
    `ml_evaluation`.
    """
    rf_results = ml_ids.train_random_forest(combined_log_path)
    if_summary = ml_ids.get_if_detection_summary()
    rb_summary = rule_based_summary or {}

    rb_dets_by_label = rb_summary.get("detections_by_label", {})
    if_dets_by_label = if_summary.get("by_class", {})
    frames_by_label = rb_summary.get("frames_seen_by_label", {})

    normal_total = int(_lookup_label(frames_by_label, 0, default=0))
    rb_fp = int(_lookup_label(rb_dets_by_label, 0, default=0))
    if_fp = int(_lookup_label(if_dets_by_label, 0, default=0))
    rb_fp_rate = (rb_fp / normal_total) if normal_total > 0 else 0.0
    if_fp_rate = (if_fp / normal_total) if normal_total > 0 else 0.0

    attack_rows = [
        ("Injection", "injection"),
        ("Fuzzing", "fuzzing"),
    ]
    if attack_simulated == "dos":
        attack_rows.append(("Bus-Off DoS", "dos"))

    print("\nPhase 5 - Hybrid IDS Evaluation")
    header = (
        f"{'Attack Class':<14} | {'Rule-Based Recall':<22} | "
        f"{'IF Recall':<22} | {'RF Confidence':<14} | "
        f"{'Test Support':<12} | Top Features"
    )
    print(header)
    print("-" * len(header))

    table_rows = []
    for display_name, key in attack_rows:
        label = ATTACK_LABEL_BY_NAME[key]
        # JSON-loaded keys may be strings; normalize to int.
        attack_total = int(_lookup_label(frames_by_label, label, default=0))
        rb_count = int(_lookup_label(rb_dets_by_label, label, default=0))
        if_count = int(_lookup_label(if_dets_by_label, label, default=0))

        rb_recall = (rb_count / attack_total) if attack_total > 0 else 0.0
        if_recall = (if_count / attack_total) if attack_total > 0 else 0.0
        rf_conf = float(rf_results.get("rf_confidence_on_true_class", {}).get(key, 0.0))
        test_support = int(rf_results.get("class_support_test", {}).get(key, 0))
        features = rf_results.get("top_features_per_attack_class", {}).get(key, FEATURE_FALLBACK)
        feature_text = ", ".join(features[:2])

        rb_text = (
            f"{rb_recall * 100:5.1f}% ({rb_count}/{attack_total})"
            if attack_total > 0
            else "n/a"
        )
        if_text = (
            f"{if_recall * 100:5.1f}% ({if_count}/{attack_total})"
            if attack_total > 0
            else "n/a"
        )

        print(
            f"{display_name:<14} | {rb_text:<22} | {if_text:<22} | "
            f"{rf_conf * 100:>6.2f}%       | {test_support:>6}       | {feature_text}"
        )
        table_rows.append(
            {
                "attack_class": key,
                "rule_based_detected": rb_count > 0,
                "rule_based_count": rb_count,
                "rule_based_recall": rb_recall,
                "if_detected": if_count > 0,
                "if_count": if_count,
                "if_recall": if_recall,
                "rf_confidence": rf_conf,
                "test_support": test_support,
                "attack_total_frames": attack_total,
                "features_that_mattered": features,
            }
        )

    trained_per_bus = if_summary.get("trained_per_bus", {})
    untrained = [bus for bus, ok in trained_per_bus.items() if not ok]
    if untrained:
        print(
            f"\n[ML] Warning: bus(es) {untrained} did not finish IF training "
            f"(insufficient baseline samples in window of {if_summary.get('baseline_seconds')}s; "
            f"need {if_summary.get('min_training_samples')} per bus). "
            f"Try a longer --ml-baseline-seconds or a lower --ml-min-train."
        )

    print(
        f"\nFalse positives on normal traffic ({normal_total} frames in the "
        f"baseline + post-attack control window): "
        f"rule-based {rb_fp_rate * 100:.3f}% ({rb_fp}), "
        f"IF {if_fp_rate * 100:.2f}% ({if_fp})"
    )
    if if_fp_rate > 0.10:
        print(
            "[ML] Note: IF false-positive rate is elevated. F1 telemetry features "
            "drift across the race even within 'normal' traffic, so a static "
            "Isolation Forest is expected to over-trigger on real data. "
            "Tightening --ml-anomaly-threshold (e.g. -0.15) trades recall for "
            "precision; --ml-baseline-seconds 90 widens the training distribution."
        )

    ml_evaluation = {
        "if_summary": if_summary,
        "rule_based_summary": rb_summary,
        "rf_results": rf_results,
        "comparison_table": table_rows,
        "attack_simulated": attack_simulated,
        "normal_traffic_summary": {
            "total_normal_frames": normal_total,
            "rule_based_false_positive_rate": rb_fp_rate,
            "rule_based_false_positives": rb_fp,
            "if_false_positive_rate": if_fp_rate,
            "if_false_positives": if_fp,
        },
    }
    _upsert_ml_evaluation(simulation_report_path, ml_evaluation)
    warnings = rf_results.get("data_quality_warnings", [])
    if warnings:
        print("\n[ML Data Quality Warnings]")
        for warning in warnings:
            print(f" - {warning}")
    return ml_evaluation


def _lookup_label(by_label: Dict[Any, Any], label: int, default: int = 0) -> int:
    if not by_label:
        return default
    if label in by_label:
        return int(by_label[label])
    if str(label) in by_label:
        return int(by_label[str(label)])
    return default


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


def run_full_ml_pipeline_from_frames(
    frames: List[Dict[str, Any]],
    attack_type: str = "injection",
    target_driver: Optional[str] = None,
    attack_start_s: float = 20.0,
    attack_duration_s: float = 20.0,
    output_dir: Optional[Path] = None,
    seed: int = 7,
    max_frames_for_ml: int = 3000,
    max_drivers_for_ml: int = 20,
    if_n_estimators: int = 100,
    rf_n_estimators: int = 200,
    if_contamination: float = 0.10,
    if_min_training_samples: int = 500,
    if_anomaly_threshold: float = -0.05,
    baseline_seconds: Optional[float] = None,
    post_attack_window_s: Optional[float] = None,
    injection_messages_per_frame: int = 4,
    fuzzing_messages_per_frame: int = 8,
) -> Dict[str, str]:
    """
    Run the streaming hybrid IDS (Rule-Based + Isolation Forest) plus the
    post-hoc Random Forest evaluation against simulation-generated traffic.

    Sampling is anchored to the attack: three contiguous full-density windows
    (baseline, attack, post-attack control) are processed, and traffic outside
    them is dropped so per-arb_id message rates and IF feature distributions
    stay consistent end-to-end. Per-driver timestamp jitter prevents
    inter-arrival-time features from collapsing to 0 when multiple drivers
    share the same arb_id.
    """
    if not frames:
        raise ValueError("No frames provided for ML simulation.")

    out_dir = Path(output_dir) if output_dir is not None else (
        Path("computed_data") / "can_security_ml" / "default_run"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_log_path = out_dir / "combined_frame_log.csv"
    report_path = out_dir / "simulation_report.json"

    attack_key = "fuzzing" if attack_type.lower() == "fuzzing" else "injection"
    first_frame_drivers = list(frames[0].get("drivers", {}).keys())
    if not first_frame_drivers:
        raise ValueError("Frames contain no driver data.")
    attack_end_s = attack_start_s + attack_duration_s

    # Pick a target that is actually present during the attack window.
    # Without this, picking purely from frame 0 can fall over when:
    #   - --ml-fast trims `max_drivers_for_ml` to 8 and frame-0's first driver
    #     happens to be ranked > 8 by the time the attack starts;
    #   - the chosen driver retires before `attack_start_s`.
    # In either case the attack would fire 0 times and RF training would crash
    # because the combined log only contains the "normal" class.
    def _drivers_present_in_attack_window() -> List[str]:
        seen: Dict[str, int] = {}
        for f in frames:
            ft = float(f.get("t", 0.0))
            if ft < attack_start_s:
                continue
            if ft > attack_end_s:
                break
            for code in f.get("drivers", {}).keys():
                seen[code] = seen.get(code, 0) + 1
        return [c for c, _ in sorted(seen.items(), key=lambda kv: -kv[1])]

    attack_window_drivers = _drivers_present_in_attack_window()
    if target_driver and target_driver in attack_window_drivers:
        actual_target = target_driver
    elif attack_window_drivers:
        actual_target = attack_window_drivers[0]
    else:
        actual_target = first_frame_drivers[0]
        print(
            f"[ML] Warning: no drivers found in attack window "
            f"[{attack_start_s:.1f}-{attack_end_s:.1f}s]; falling back to "
            f"frame-0 driver '{actual_target}'."
        )

    # Auto-size baseline to a bounded window that ends right before the attack.
    # Anchoring it to the run-up of the attack (not the first N seconds of the
    # race) means the IF and the rule-based frequency rule both train on
    # representative racing telemetry, and avoids the sampling-density mismatch
    # that happens when a sparse early-race baseline is compared against a
    # dense attack window.
    if baseline_seconds is None:
        baseline_seconds = min(
            DEFAULT_BASELINE_DURATION_S,
            max(1.0, attack_start_s - 1.0),
        )
    # The post-attack control window is what IF/RB false-positive rate is
    # measured against. Anchored adjacent to the attack so feature
    # distributions match the baseline; everything past this window is
    # discarded so a minor data drift hours later does not drown the result.
    if post_attack_window_s is None:
        post_attack_window_s = max(DEFAULT_POST_ATTACK_WINDOW_S, attack_duration_s)

    rng = random.Random(seed)
    ml_ids = MLIDS(
        baseline_seconds=baseline_seconds,
        if_n_estimators=if_n_estimators,
        rf_n_estimators=rf_n_estimators,
        if_contamination=if_contamination,
        min_training_samples=if_min_training_samples,
        anomaly_threshold=if_anomaly_threshold,
        models_dir=out_dir / "models",
        alerts_log_path=out_dir / "ml_alerts.log",
    )
    rule_ids = RuleBasedIDS(baseline_seconds=baseline_seconds)

    baseline_window_start = max(0.0, attack_start_s - baseline_seconds)
    post_attack_end = attack_end_s + post_attack_window_s
    sampled_frames = _sample_frames(
        frames=frames,
        baseline_window_start=baseline_window_start,
        attack_start_s=attack_start_s,
        attack_end_s=attack_end_s,
        post_attack_end=post_attack_end,
    )
    n_attack = sum(
        1 for f in sampled_frames if attack_start_s <= float(f.get("t", 0.0)) <= attack_end_s
    )
    n_baseline = sum(
        1
        for f in sampled_frames
        if baseline_window_start <= float(f.get("t", 0.0)) < attack_start_s
    )
    n_post = sum(
        1
        for f in sampled_frames
        if attack_end_s < float(f.get("t", 0.0)) <= post_attack_end
    )
    print(
        f"[ML] Sampled {len(sampled_frames)} frames "
        f"(original={len(frames)}, baseline [{baseline_window_start:.1f}-"
        f"{attack_start_s:.1f}s]={n_baseline}, attack "
        f"[{attack_start_s:.1f}-{attack_end_s:.1f}s]={n_attack}, post-attack "
        f"[{attack_end_s:.1f}-{post_attack_end:.1f}s]={n_post})..."
    )

    rows: List[Dict[str, Any]] = []
    attack_messages_emitted = 0
    for idx, frame in enumerate(sampled_frames, start=1):
        t = float(frame.get("t", 0.0))
        drivers = frame.get("drivers", {})
        if max_drivers_for_ml > 0 and len(drivers) > max_drivers_for_ml:
            # Always preserve the attack target in the trimmed set; otherwise
            # `--ml-fast` (which sets max_drivers_for_ml=8) can drop the target
            # and the attack never fires.
            items = list(drivers.items())
            if actual_target in drivers:
                others = [(k, v) for k, v in items if k != actual_target]
                kept = [(actual_target, drivers[actual_target])] + others[: max_drivers_for_ml - 1]
                drivers = dict(kept)
            else:
                drivers = dict(items[:max_drivers_for_ml])
        n_drivers = len(drivers)
        driver_jitter = (
            min(DRIVER_JITTER_S, MAX_FRAME_JITTER_SPAN_S / max(1, n_drivers))
            if n_drivers > 0
            else DRIVER_JITTER_S
        )
        messages: List[Dict[str, Any]] = []

        for driver_idx, (code, pos) in enumerate(drivers.items()):
            t_driver = t + driver_idx * driver_jitter
            messages.extend(_build_baseline_messages(t_driver, code, pos))

        if attack_start_s <= t <= attack_end_s and actual_target in drivers:
            attack_t_base = t + n_drivers * driver_jitter
            if attack_key == "fuzzing":
                attack_msgs = _fuzzing_messages(
                    attack_t_base,
                    actual_target,
                    rng,
                    count=fuzzing_messages_per_frame,
                )
            else:
                attack_msgs = _injection_messages(
                    attack_t_base,
                    actual_target,
                    count=injection_messages_per_frame,
                )
            messages.extend(attack_msgs)
            attack_messages_emitted += len(attack_msgs)

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
            label = int(msg["label"])
            ml_ids.process_frame(msg["bus"], msg, label=label)
            rule_ids.process_frame(
                timestamp=msg["timestamp"],
                bus=msg["bus"],
                arb_id=msg["arb_id"],
                payload=msg["payload"],
                label=label,
            )
        if idx % 250 == 0:
            print(f"[ML] Processed {idx}/{len(sampled_frames)} sampled frames...")

    # Flush any pending feature vectors to score the final batch on each bus.
    ml_ids.finalize()

    import pandas as pd

    pd.DataFrame(rows).to_csv(combined_log_path, index=False)
    print(
        f"[ML] Wrote combined log with {len(rows)} CAN messages "
        f"({attack_messages_emitted} attack messages on driver '{actual_target}'): "
        f"{combined_log_path}"
    )
    if attack_messages_emitted == 0:
        raise RuntimeError(
            f"No attack messages were generated for target '{actual_target}' in "
            f"window [{attack_start_s}-{attack_end_s}s]. RF training requires at "
            f"least two classes. Check --attack-target or --ml-attack-start."
        )

    run_ml_evaluation_phase(
        ml_ids=ml_ids,
        combined_log_path=combined_log_path,
        simulation_report_path=report_path,
        rule_based_summary=rule_ids.summary(),
        attack_simulated=attack_key,
    )
    print(f"[ML] Wrote simulation report: {report_path}")

    return {
        "combined_log_path": str(combined_log_path),
        "simulation_report_path": str(report_path),
        "alerts_log_path": str(out_dir / "ml_alerts.log"),
        "models_dir": str(out_dir / "models"),
    }


DEFAULT_BASELINE_DURATION_S = 60.0
DEFAULT_POST_ATTACK_WINDOW_S = 60.0


def _sample_frames(
    frames: List[Dict[str, Any]],
    baseline_window_start: float,
    attack_start_s: float,
    attack_end_s: float,
    post_attack_end: float,
) -> List[Dict[str, Any]]:
    """
    Pick the frames the IDS pipeline runs on.

    The selection is anchored to the attack: we keep three full-density
    contiguous windows so per-arb_id message rates and IF feature
    distributions are stable end-to-end:

      - baseline window     `[baseline_window_start, attack_start_s)`
      - attack window       `[attack_start_s, attack_end_s]`
      - post-attack control `(attack_end_s, post_attack_end]`

    Everything outside these windows (pre-baseline early-race traffic and
    far-future frames) is dropped. This avoids two real-data pathologies:

      1. Stride-sampling a long pre-attack period made the rule-based
         frequency rule see a 10x rate jump at attack time and fire on
         every normal message in the attack window.
      2. Scoring frames from minute 12+ of a 90-minute race against a
         60-second baseline produced a 38% IF false-positive rate due to
         legitimate feature drift across the race.
    """
    if not frames:
        return []

    kept: List[Dict[str, Any]] = []
    for f in frames:
        t = float(f.get("t", 0.0))
        if t < baseline_window_start:
            continue
        if t > post_attack_end:
            continue
        kept.append(f)

    kept.sort(key=lambda f: float(f.get("t", 0.0)))
    return kept


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


def _injection_messages(
    timestamp: float,
    _target_driver: str,
    count: int = 4,
) -> List[Dict[str, Any]]:
    forged_speed_kph = 380.0
    return [
        _mk_msg(
            timestamp + i * ATTACK_JITTER_S,
            "sensor_bus",
            0x100,
            _encode_u16(forged_speed_kph, 10.0),
            ATTACK_LABEL_BY_NAME["injection"],
        )
        for i in range(max(1, int(count)))
    ]


def _fuzzing_messages(
    timestamp: float,
    _target_driver: str,
    rng: random.Random,
    count: int = 8,
) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    for i in range(max(1, int(count))):
        fuzz_id = rng.randint(0x200, 0x7FF)
        payload_len = rng.randint(2, 8)
        payload = bytes(rng.randint(0, 255) for _ in range(payload_len))
        msgs.append(
            _mk_msg(
                timestamp + i * ATTACK_JITTER_S,
                "powertrain_bus",
                fuzz_id,
                payload,
                ATTACK_LABEL_BY_NAME["fuzzing"],
            )
        )
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
