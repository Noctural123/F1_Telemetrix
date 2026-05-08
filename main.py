from src.f1_data import get_race_telemetry, enable_cache, get_circuit_rotation, load_session, get_quali_telemetry, list_rounds, list_sprints
from src.arcade_replay import run_arcade_replay
from src.can_security import (
    build_security_overlay,
    evaluate_random_forest_cross_run,
    run_full_ml_pipeline_from_frames,
)

from src.interfaces.qualifying import run_qualifying_replay
import json
import sys
from datetime import datetime
from pathlib import Path

def main(year=None, round_number=None, playback_speed=1, session_type='R'):
  print(f"Loading F1 {year} Round {round_number} Session '{session_type}'")
  session = load_session(year, round_number, session_type)

  print(f"Loaded session: {session.event['EventName']} - {session.event['RoundNumber']} - {session_type}")

  # Enable cache for fastf1
  enable_cache()

  if session_type == 'Q' or session_type == 'SQ':

    # Get the drivers who participated and their lap times

    qualifying_session_data = get_quali_telemetry(session, session_type=session_type)

    # Run the arcade screen showing qualifying results

    title = f"{session.event['EventName']} - {'Sprint Qualifying' if session_type == 'SQ' else 'Qualifying Results'}"
    
    run_qualifying_replay(
      session=session,
      data=qualifying_session_data,
      title=title,
    )

  else:

    # Get the drivers who participated in the race

    race_telemetry = get_race_telemetry(session, session_type=session_type)
    ml_only = "--ml-only" in sys.argv

    # Get example lap for track layout
    # Qualifying lap preferred for DRS zones (fallback to fastest race lap (no DRS data))
    example_lap = None
    if not ml_only:
        try:
            print("Attempting to load qualifying session for track layout...")
            quali_session = load_session(year, round_number, 'Q')
            if quali_session is not None and len(quali_session.laps) > 0:
                fastest_quali = quali_session.laps.pick_fastest()
                if fastest_quali is not None:
                    quali_telemetry = fastest_quali.get_telemetry()
                    if 'DRS' in quali_telemetry.columns:
                        example_lap = quali_telemetry
                        print(f"Using qualifying lap from driver {fastest_quali['Driver']} for DRS Zones")
        except Exception as e:
            print(f"Could not load qualifying session: {e}")

    # fallback: Use fastest race lap
    if example_lap is None and not ml_only:
        fastest_lap = session.laps.pick_fastest()
        if fastest_lap is not None:
            example_lap = fastest_lap.get_telemetry()
            print("Using fastest race lap (DRS detection may use speed-based fallback)")
        else:
            print("Error: No valid laps found in session")
            return

    drivers = session.drivers

    # Get circuit rotation

    circuit_rotation = get_circuit_rotation(session)

    # Run the arcade replay

    # Check for optional chart flag
    chart = "--chart" in sys.argv
    security_demo = "--security-demo" in sys.argv
    attack_type = "fuzzing" if "--attack-fuzzing" in sys.argv else "injection"
    batch_runs = 1
    max_frames_for_ml = 3000
    max_drivers_for_ml = 20
    if_n_estimators = 100
    rf_n_estimators = 200
    if_contamination = 0.10
    if_min_training_samples = 500
    if_anomaly_threshold = -0.05
    # Default to None so the pipeline auto-sizes the baseline to cover all
    # pre-attack telemetry (best for representative IF training on real F1
    # data). Override with --ml-baseline-seconds for a strict streaming demo.
    baseline_seconds = None
    injection_messages_per_frame = 4
    ml_fast = "--ml-fast" in sys.argv
    if ml_fast:
        max_frames_for_ml = 1500
        max_drivers_for_ml = 8
        if_n_estimators = 30
        rf_n_estimators = 60
        if_min_training_samples = 200
    if "--ml-batch-runs" in sys.argv:
        runs_idx = sys.argv.index("--ml-batch-runs") + 1
        if runs_idx < len(sys.argv):
            try:
                batch_runs = max(1, int(sys.argv[runs_idx]))
            except ValueError:
                print("Invalid value for --ml-batch-runs. Using 1.")
                batch_runs = 1
    if "--ml-max-frames" in sys.argv:
        frames_idx = sys.argv.index("--ml-max-frames") + 1
        if frames_idx < len(sys.argv):
            try:
                max_frames_for_ml = max(100, int(sys.argv[frames_idx]))
            except ValueError:
                print("Invalid value for --ml-max-frames. Using default.")
    if "--ml-max-drivers" in sys.argv:
        drivers_idx = sys.argv.index("--ml-max-drivers") + 1
        if drivers_idx < len(sys.argv):
            try:
                max_drivers_for_ml = max(2, int(sys.argv[drivers_idx]))
            except ValueError:
                print("Invalid value for --ml-max-drivers. Using default.")
    if "--ml-if-estimators" in sys.argv:
        if_idx = sys.argv.index("--ml-if-estimators") + 1
        if if_idx < len(sys.argv):
            try:
                if_n_estimators = max(10, int(sys.argv[if_idx]))
            except ValueError:
                print("Invalid value for --ml-if-estimators. Using default.")
    if "--ml-rf-estimators" in sys.argv:
        rf_idx = sys.argv.index("--ml-rf-estimators") + 1
        if rf_idx < len(sys.argv):
            try:
                rf_n_estimators = max(20, int(sys.argv[rf_idx]))
            except ValueError:
                print("Invalid value for --ml-rf-estimators. Using default.")
    if "--ml-contamination" in sys.argv:
        c_idx = sys.argv.index("--ml-contamination") + 1
        if c_idx < len(sys.argv):
            try:
                if_contamination = max(0.001, min(0.5, float(sys.argv[c_idx])))
            except ValueError:
                print("Invalid value for --ml-contamination. Using default.")
    if "--ml-min-train" in sys.argv:
        mt_idx = sys.argv.index("--ml-min-train") + 1
        if mt_idx < len(sys.argv):
            try:
                if_min_training_samples = max(20, int(sys.argv[mt_idx]))
            except ValueError:
                print("Invalid value for --ml-min-train. Using default.")
    if "--ml-anomaly-threshold" in sys.argv:
        at_idx = sys.argv.index("--ml-anomaly-threshold") + 1
        if at_idx < len(sys.argv):
            try:
                if_anomaly_threshold = float(sys.argv[at_idx])
            except ValueError:
                print("Invalid value for --ml-anomaly-threshold. Using default.")
    if "--ml-baseline-seconds" in sys.argv:
        bs_idx = sys.argv.index("--ml-baseline-seconds") + 1
        if bs_idx < len(sys.argv):
            try:
                baseline_seconds = max(0.5, float(sys.argv[bs_idx]))
            except ValueError:
                print("Invalid value for --ml-baseline-seconds. Using default.")
    if "--ml-injection-count" in sys.argv:
        ic_idx = sys.argv.index("--ml-injection-count") + 1
        if ic_idx < len(sys.argv):
            try:
                injection_messages_per_frame = max(1, int(sys.argv[ic_idx]))
            except ValueError:
                print("Invalid value for --ml-injection-count. Using default.")
    attack_start_s = 20.0
    attack_duration_s = 20.0
    if "--ml-attack-start" in sys.argv:
        as_idx = sys.argv.index("--ml-attack-start") + 1
        if as_idx < len(sys.argv):
            try:
                attack_start_s = max(1.0, float(sys.argv[as_idx]))
            except ValueError:
                print("Invalid value for --ml-attack-start. Using default.")
    if "--ml-attack-duration" in sys.argv:
        ad_idx = sys.argv.index("--ml-attack-duration") + 1
        if ad_idx < len(sys.argv):
            try:
                attack_duration_s = max(1.0, float(sys.argv[ad_idx]))
            except ValueError:
                print("Invalid value for --ml-attack-duration. Using default.")
    post_attack_window_s = None
    if "--ml-post-attack-window" in sys.argv:
        pw_idx = sys.argv.index("--ml-post-attack-window") + 1
        if pw_idx < len(sys.argv):
            try:
                post_attack_window_s = max(1.0, float(sys.argv[pw_idx]))
            except ValueError:
                print("Invalid value for --ml-post-attack-window. Using default.")
    attack_target = None
    if "--attack-target" in sys.argv:
        target_index = sys.argv.index("--attack-target") + 1
        if target_index < len(sys.argv):
            attack_target = sys.argv[target_index].upper()
    security_overlay = None
    if security_demo:
        print(f"Security demo enabled ({attack_type}).")
        security_overlay = build_security_overlay(
            race_telemetry["frames"],
            attack_type=attack_type,
            target_driver=attack_target,
            attack_start_s=attack_start_s,
            attack_duration_s=attack_duration_s,
        )
        if "--ml-eval" in sys.argv:
            print(f"Running ML IDS evaluation pipeline ({batch_runs} run(s))...")
            run_root = Path("computed_data") / "can_security_ml" / "runs"
            run_root.mkdir(parents=True, exist_ok=True)
            manifest_path = run_root / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    manifest = {"runs": []}
            else:
                manifest = {"runs": []}

            for run_num in range(1, batch_runs + 1):
                run_name = (
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{attack_type}_run_{run_num:03d}"
                    if batch_runs > 1
                    else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{attack_type}"
                )
                output_dir = run_root / run_name
                print(f"\n[ML Batch] Starting run {run_num}/{batch_runs}: {run_name}")
                print(
                    f"[ML Batch] Settings: frames={max_frames_for_ml}, "
                    f"drivers={max_drivers_for_ml}, IF trees={if_n_estimators}, RF trees={rf_n_estimators}"
                )
                outputs = run_full_ml_pipeline_from_frames(
                    race_telemetry["frames"],
                    attack_type=attack_type,
                    target_driver=attack_target,
                    attack_start_s=attack_start_s,
                    attack_duration_s=attack_duration_s,
                    output_dir=output_dir,
                    seed=7 + run_num,
                    max_frames_for_ml=max_frames_for_ml,
                    max_drivers_for_ml=max_drivers_for_ml,
                    if_n_estimators=if_n_estimators,
                    rf_n_estimators=rf_n_estimators,
                    if_contamination=if_contamination,
                    if_min_training_samples=if_min_training_samples,
                    if_anomaly_threshold=if_anomaly_threshold,
                    baseline_seconds=baseline_seconds,
                    post_attack_window_s=post_attack_window_s,
                    injection_messages_per_frame=injection_messages_per_frame,
                )
                print(f"ML combined log: {outputs['combined_log_path']}")
                print(f"ML report: {outputs['simulation_report_path']}")
                print(f"ML alerts: {outputs['alerts_log_path']}")
                print(f"ML models dir: {outputs['models_dir']}")

                manifest["runs"].append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "attack_type": attack_type,
                        "run_name": run_name,
                        "combined_log_path": outputs["combined_log_path"],
                        "simulation_report_path": outputs["simulation_report_path"],
                    }
                )

            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"\n[ML Batch] Saved manifest: {manifest_path}")

            if "--ml-cross-run-eval" in sys.argv and batch_runs >= 2:
                cross_run_logs = [
                    Path(entry["combined_log_path"])
                    for entry in manifest["runs"][-batch_runs:]
                    if entry["attack_type"] == attack_type
                ]
                if len(cross_run_logs) >= 2:
                    holdout = max(1, len(cross_run_logs) // 5)  # ~20% test
                    train_paths = cross_run_logs[:-holdout]
                    test_paths = cross_run_logs[-holdout:]
                    cross_report_path = run_root / (
                        f"cross_run_eval_{attack_type}_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    print(
                        f"\n[ML Cross-Run] Training on {len(train_paths)} run(s), "
                        f"testing on {len(test_paths)} held-out run(s)."
                    )
                    cross_result = evaluate_random_forest_cross_run(
                        train_csv_paths=train_paths,
                        test_csv_paths=test_paths,
                        rf_n_estimators=rf_n_estimators,
                        output_path=cross_report_path,
                    )
                    print(
                        f"[ML Cross-Run] Test rows={cross_result['n_test_rows']}, "
                        f"overall accuracy={cross_result['overall_accuracy']:.4f}"
                    )
                    for class_name, conf in cross_result["rf_confidence_on_true_class"].items():
                        if conf > 0:
                            print(
                                f"[ML Cross-Run]  {class_name}: confidence-on-true-class = {conf:.4f}"
                            )
                    print(f"[ML Cross-Run] Saved report: {cross_report_path}")

    if ml_only and "--ml-eval" in sys.argv:
        print("ML-only mode enabled; skipping Arcade replay window.")
        return

    run_arcade_replay(
      frames=race_telemetry['frames'],
      track_statuses=race_telemetry['track_statuses'],
      example_lap=example_lap,
      drivers=drivers,
      playback_speed=playback_speed,
      driver_colors=race_telemetry['driver_colors'],
      title=f"{session.event['EventName']} - {'Sprint' if session_type == 'S' else 'Race'}",
      total_laps=race_telemetry['total_laps'],
      circuit_rotation=circuit_rotation,
      chart=chart,
      security_overlay=security_overlay,
    )

if __name__ == "__main__":

  # Get the year and round number from user input

  if "--year" in sys.argv:
    year_index = sys.argv.index("--year") + 1
    year = int(sys.argv[year_index])
  else:
    year = 2025  # Default year

  if "--round" in sys.argv:
    round_index = sys.argv.index("--round") + 1
    round_number = int(sys.argv[round_index])
  else:
    round_number = 12  # Default round number

  if "--list-rounds" in sys.argv:
    list_rounds(year)
  elif "--list-sprints" in sys.argv:
    list_sprints(year)
  else:

    playback_speed = 1

    # Session type selection
    session_type = 'SQ' if "--sprint-qualifying" in sys.argv else ('S' if "--sprint" in sys.argv else ('Q' if "--qualifying" in sys.argv else 'R'))
    
    main(year, round_number, playback_speed, session_type=session_type)