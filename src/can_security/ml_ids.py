from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BUS_NAMES = ("sensor_bus", "display_bus", "powertrain_bus")
FEATURE_COLUMNS = [
    "message_rate",
    "inter_arrival_time",
    "payload_byte_variance",
    "payload_mean",
    "arb_id_frequency",
]
ATTACK_NAME_BY_LABEL = {0: "normal", 1: "injection", 2: "fuzzing", 3: "dos"}


@dataclass
class Layer1Detection:
    timestamp: float
    bus: str
    arb_id: int
    anomaly_score: float
    phase: str
    features: Dict[str, float]
    label: int = 0


class _BusState:
    def __init__(
        self,
        baseline_seconds: float = 10.0,
        min_training_samples: int = 500,
    ) -> None:
        self.baseline_seconds = float(baseline_seconds)
        self.min_training_samples = int(min_training_samples)
        self.first_frame_ts: Optional[float] = None
        self.frame_count = 0
        self.arb_counter: Counter = Counter()
        self.recent_timestamps: Dict[int, Deque[float]] = {}
        self.last_arb_timestamp: Dict[int, float] = {}
        self.training_buffer: List[List[float]] = []
        self.trained = False
        self.model: Optional[IsolationForest] = None
        self.lock = threading.Lock()

    def phase(self, current_ts: Optional[float] = None) -> str:
        if self.first_frame_ts is None or current_ts is None:
            return "baseline"
        return (
            "baseline"
            if (current_ts - self.first_frame_ts) < self.baseline_seconds
            else "attack"
        )


class MLIDS:
    """
    Two-layer ML IDS:
    1) Isolation Forest per-bus real-time anomaly detection
    2) Random Forest post-simulation attack classification
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        alerts_log_path: Optional[Path] = None,
        baseline_seconds: float = 10.0,
        if_n_estimators: int = 100,
        if_contamination: float = 0.10,
        rf_n_estimators: int = 200,
        min_training_samples: int = 500,
        anomaly_threshold: float = -0.05,
        batch_score_size: int = 512,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        self.models_dir = models_dir or (base_dir / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_log_path = alerts_log_path or (base_dir / "ml_alerts.log")
        self._configure_alert_logger()

        self._states: Dict[str, _BusState] = {
            bus: _BusState(
                baseline_seconds=baseline_seconds,
                min_training_samples=min_training_samples,
            )
            for bus in BUS_NAMES
        }
        self._detections: Dict[str, List[Layer1Detection]] = {bus: [] for bus in BUS_NAMES}
        self._pending_per_bus: Dict[str, List[Dict[str, Any]]] = {bus: [] for bus in BUS_NAMES}
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self.if_n_estimators = int(if_n_estimators)
        self.if_contamination = float(if_contamination)
        self.rf_n_estimators = int(rf_n_estimators)
        self.min_training_samples = int(min_training_samples)
        self.anomaly_threshold = float(anomaly_threshold)
        self.baseline_seconds = float(baseline_seconds)
        self.batch_score_size = max(1, int(batch_score_size))

    def _configure_alert_logger(self) -> None:
        # The "ml_ids_alerts" logger is module-level; multiple MLIDS instances
        # in the same process (e.g. one per batch run) must each redirect it
        # to their own alerts log, so we always swap handlers here.
        self.alert_logger = logging.getLogger("ml_ids_alerts")
        self.alert_logger.setLevel(logging.INFO)
        self.alert_logger.propagate = False
        for h in list(self.alert_logger.handlers):
            self.alert_logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        Path(self.alerts_log_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.alerts_log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.alert_logger.addHandler(handler)

    @staticmethod
    def _extract_frame_fields(frame: Any) -> Tuple[float, int, List[int]]:
        if isinstance(frame, dict):
            timestamp = float(frame.get("timestamp", time.time()))
            arb_id = int(frame.get("arb_id", frame.get("can_id", 0)))
            payload = frame.get("payload", [])
        else:
            timestamp = float(getattr(frame, "timestamp", time.time()))
            arb_id = int(getattr(frame, "arb_id", getattr(frame, "can_id", 0)))
            payload = getattr(frame, "payload", [])

        if isinstance(payload, str):
            raw = payload.strip().replace(" ", "")
            if raw.startswith("0x"):
                raw = raw[2:]
            payload_bytes = list(bytes.fromhex(raw)) if raw else []
        elif isinstance(payload, (bytes, bytearray)):
            payload_bytes = list(payload)
        else:
            payload_bytes = [int(x) for x in payload]

        payload_bytes = (payload_bytes[:8] + [0] * 8)[:8]
        return timestamp, arb_id, payload_bytes

    @staticmethod
    def _compute_payload_stats(payload: List[int]) -> Tuple[float, float]:
        arr = np.array(payload, dtype=float)
        return float(arr.var()), float(arr.mean())

    def _extract_features(self, bus: str, timestamp: float, arb_id: int, payload: List[int]) -> Dict[str, float]:
        state = self._states[bus]
        with state.lock:
            state.frame_count += 1
            state.arb_counter[arb_id] += 1

            bucket = state.recent_timestamps.setdefault(arb_id, deque())
            bucket.append(timestamp)
            while bucket and (timestamp - bucket[0]) > 1.0:
                bucket.popleft()
            message_rate = float(len(bucket))

            last_ts = state.last_arb_timestamp.get(arb_id)
            inter_arrival_ms = 0.0 if last_ts is None else max((timestamp - last_ts) * 1000.0, 0.0)
            state.last_arb_timestamp[arb_id] = timestamp

            payload_var, payload_mean = self._compute_payload_stats(payload)
            arb_id_frequency = float(state.arb_counter[arb_id]) / float(max(state.frame_count, 1))

        return {
            "message_rate": message_rate,
            "inter_arrival_time": inter_arrival_ms,
            "payload_byte_variance": payload_var,
            "payload_mean": payload_mean,
            "arb_id_frequency": arb_id_frequency,
        }

    def process_frame(
        self,
        bus: str,
        frame: Any,
        label: int = 0,
    ) -> Optional[Layer1Detection]:
        if bus not in self._states:
            return None

        timestamp, arb_id, payload = self._extract_frame_fields(frame)
        features = self._extract_features(bus, timestamp, arb_id, payload)
        state = self._states[bus]
        feature_vector = [features[name] for name in FEATURE_COLUMNS]

        do_drain = False
        with state.lock:
            if state.first_frame_ts is None:
                state.first_frame_ts = timestamp
            phase = state.phase(timestamp)

            if not state.trained:
                if phase == "baseline":
                    state.training_buffer.append(feature_vector)
                    return None
                if len(state.training_buffer) < state.min_training_samples:
                    # Past baseline window but still not enough training data;
                    # skip rather than fit a too-small model.
                    return None
                model = IsolationForest(
                    n_estimators=self.if_n_estimators,
                    contamination=self.if_contamination,
                    random_state=42,
                )
                model.fit(np.array(state.training_buffer))
                state.model = model
                state.trained = True

            self._pending_per_bus[bus].append(
                {
                    "timestamp": timestamp,
                    "bus": bus,
                    "arb_id": arb_id,
                    "phase": phase,
                    "features": features,
                    "feature_vector": feature_vector,
                    "label": int(label),
                }
            )
            if len(self._pending_per_bus[bus]) >= self.batch_score_size:
                do_drain = True

        if do_drain:
            return self._drain_pending(bus)
        return None

    def _drain_pending(self, bus: str) -> Optional[Layer1Detection]:
        state = self._states[bus]
        with state.lock:
            pending = self._pending_per_bus[bus]
            self._pending_per_bus[bus] = []
            model = state.model
        if not pending or model is None:
            return None
        features_arr = np.asarray([p["feature_vector"] for p in pending], dtype=float)
        scores = model.decision_function(features_arr)
        threshold = self.anomaly_threshold
        last_detection: Optional[Layer1Detection] = None
        for entry, score in zip(pending, scores):
            score_f = float(score)
            if score_f < threshold:
                detection = Layer1Detection(
                    timestamp=entry["timestamp"],
                    bus=entry["bus"],
                    arb_id=entry["arb_id"],
                    anomaly_score=score_f,
                    phase=entry["phase"],
                    features=entry["features"],
                    label=int(entry["label"]),
                )
                self._detections[bus].append(detection)
                self.alert_logger.info(
                    json.dumps(
                        {
                            "timestamp": entry["timestamp"],
                            "bus": entry["bus"],
                            "arb_id": entry["arb_id"],
                            "anomaly_score": score_f,
                            "features": entry["features"],
                            "phase": entry["phase"],
                            "label": int(entry["label"]),
                        }
                    )
                )
                last_detection = detection
        return last_detection

    def finalize(self) -> None:
        """Flush any pending feature vectors for end-of-pipeline scoring."""
        for bus in BUS_NAMES:
            self._drain_pending(bus)

    def _monitor_bus(self, bus_name: str, bus: Any, poll_timeout: float = 0.1) -> None:
        while not self._stop_event.is_set():
            frame = None
            try:
                frame = bus.recv(timeout=poll_timeout)
            except TypeError:
                frame = bus.recv()
            except Exception:
                continue
            if frame is not None:
                # Realtime path has no ground-truth label, default to 0.
                self.process_frame(bus_name, frame, label=0)

    def start_realtime_monitoring(self, buses: Dict[str, Any]) -> None:
        self._stop_event.clear()
        self._threads = []
        for bus_name in BUS_NAMES:
            bus_obj = buses.get(bus_name)
            if bus_obj is None:
                continue
            t = threading.Thread(target=self._monitor_bus, args=(bus_name, bus_obj), daemon=True)
            t.start()
            self._threads.append(t)

    def stop_realtime_monitoring(self, join_timeout: float = 1.0) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=join_timeout)
        self._threads = []
        # Drain any post-baseline frames that were buffered for batch scoring.
        self.finalize()

    def get_if_detection_summary(self) -> Dict[str, Any]:
        by_class: Dict[int, int] = defaultdict(int)
        for bus_dets in self._detections.values():
            for det in bus_dets:
                by_class[int(det.label)] += 1
        trained_status = {bus: bool(self._states[bus].trained) for bus in BUS_NAMES}
        return {
            "bus_counts": {bus: len(rows) for bus, rows in self._detections.items()},
            "total_anomalies": sum(len(rows) for rows in self._detections.values()),
            "by_class": {int(k): int(v) for k, v in by_class.items()},
            "trained_per_bus": trained_status,
            "anomaly_threshold": self.anomaly_threshold,
            "if_contamination": self.if_contamination,
            "min_training_samples": self.min_training_samples,
            "baseline_seconds": self.baseline_seconds,
        }

    @staticmethod
    def _extract_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["timestamp"] = pd.to_numeric(data["timestamp"], errors="coerce")
        data["arb_id"] = data["arb_id"].apply(lambda v: int(str(v), 0) if isinstance(v, str) else int(v))
        data = data.sort_values(["bus", "timestamp"]).reset_index(drop=True)

        payload_stats = data["payload"].apply(_payload_to_var_mean)
        data["payload_byte_variance"] = [v[0] for v in payload_stats]
        data["payload_mean"] = [v[1] for v in payload_stats]

        data["message_rate"] = (
            data.groupby(["bus", "arb_id"])["timestamp"]
            .transform(lambda s: s.rolling("1s", on=None).count() if isinstance(s.index, pd.DatetimeIndex) else _rolling_count_1s(s))
            .astype(float)
        )
        data["inter_arrival_time"] = (
            data.groupby(["bus", "arb_id"])["timestamp"].diff().fillna(0.0).clip(lower=0.0) * 1000.0
        )

        bus_total = data.groupby("bus").cumcount() + 1
        per_arb = data.groupby(["bus", "arb_id"]).cumcount() + 1
        data["arb_id_frequency"] = per_arb / bus_total
        return data

    def train_random_forest(self, combined_frame_log_path: Path) -> Dict[str, Any]:
        csv_path = Path(combined_frame_log_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Combined log not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"timestamp", "bus", "arb_id", "payload", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Combined log missing columns: {sorted(missing)}")

        features_df = self._extract_features_from_df(df)
        X = features_df[FEATURE_COLUMNS].fillna(0.0)
        y = features_df["label"].astype(int)
        if y.nunique() < 2:
            raise ValueError("Need at least two classes in the combined log for Random Forest training.")

        class_counts = y.value_counts()
        can_stratify = y.nunique() > 1 and int(class_counts.min()) >= 2
        if not can_stratify:
            print(
                "[RF] Low-count class detected; using non-stratified train/test split for this run."
            )
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if can_stratify else None,
        )
        clf = RandomForestClassifier(n_estimators=self.rf_n_estimators, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        labels_sorted = sorted(y.unique().tolist())
        report = classification_report(
            y_test,
            y_pred,
            labels=labels_sorted,
            target_names=[ATTACK_NAME_BY_LABEL.get(i, str(i)) for i in labels_sorted],
            output_dict=True,
            zero_division=0,
        )
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist()

        importances = clf.feature_importances_
        top3 = [FEATURE_COLUMNS[i] for i in np.argsort(importances)[::-1][:3]]
        top_features_per_class = {
            ATTACK_NAME_BY_LABEL.get(label, str(label)): top3 for label in labels_sorted if label != 0
        }

        model_path = self.models_dir / "random_forest.pkl"
        eval_path = self.models_dir / "rf_evaluation.json"
        joblib.dump(clf, model_path)

        class_metrics = {
            ATTACK_NAME_BY_LABEL.get(lbl, str(lbl)): {
                "precision": float(report[ATTACK_NAME_BY_LABEL.get(lbl, str(lbl))]["precision"]),
                "recall": float(report[ATTACK_NAME_BY_LABEL.get(lbl, str(lbl))]["recall"]),
                "f1_score": float(report[ATTACK_NAME_BY_LABEL.get(lbl, str(lbl))]["f1-score"]),
            }
            for lbl in labels_sorted
        }

        support_total_counts = y.value_counts().to_dict()
        support_test_counts = y_test.value_counts().to_dict()
        class_support_total = {
            ATTACK_NAME_BY_LABEL.get(lbl, str(lbl)): int(support_total_counts.get(lbl, 0)) for lbl in labels_sorted
        }
        class_support_test = {
            ATTACK_NAME_BY_LABEL.get(lbl, str(lbl)): int(support_test_counts.get(lbl, 0)) for lbl in labels_sorted
        }

        avg_conf = {}
        conf_true_class = {}
        proba_columns = [ATTACK_NAME_BY_LABEL.get(c, str(c)) for c in clf.classes_]
        for class_name in ["injection", "fuzzing", "dos"]:
            if class_name in proba_columns:
                idx = proba_columns.index(class_name)
                avg_conf[class_name] = float(np.mean(y_prob[:, idx]))
                class_label = clf.classes_[idx]
                mask = y_test.to_numpy() == class_label
                conf_true_class[class_name] = float(np.mean(y_prob[mask, idx])) if np.any(mask) else 0.0
            else:
                avg_conf[class_name] = 0.0
                conf_true_class[class_name] = 0.0

        low_support_warnings = []
        for class_name in ["injection", "fuzzing", "dos"]:
            support = int(class_support_test.get(class_name, 0))
            if support == 0:
                low_support_warnings.append(
                    f"Class '{class_name}' has no test samples; metrics/confidence are not meaningful."
                )
            elif support < 20:
                low_support_warnings.append(
                    f"Class '{class_name}' has very low test support ({support}); metrics are unstable."
                )

        evaluation = {
            "overall_accuracy": acc,
            "per_class_metrics": class_metrics,
            "class_support_total": class_support_total,
            "class_support_test": class_support_test,
            "top_features_per_attack_class": top_features_per_class,
            "confusion_matrix": {
                "labels": [ATTACK_NAME_BY_LABEL.get(i, str(i)) for i in labels_sorted],
                "matrix": cm,
            },
            "rf_confidence": avg_conf,
            "rf_confidence_on_true_class": conf_true_class,
            "data_quality_warnings": low_support_warnings,
            "model_path": str(model_path),
        }

        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2)

        print("\n[Random Forest Evaluation]")
        print(f"Overall accuracy: {acc:.4f}")
        print("Per-class metrics:")
        for class_name, metrics in class_metrics.items():
            print(
                f" - {class_name}: precision={metrics['precision']:.3f}, "
                f"recall={metrics['recall']:.3f}, f1={metrics['f1_score']:.3f}"
            )
        print("Class support (total/test):")
        for class_name in class_support_total:
            print(
                f" - {class_name}: total={class_support_total[class_name]}, "
                f"test={class_support_test.get(class_name, 0)}"
            )
        print("Top 3 most important features per attack class:")
        for class_name, feats in top_features_per_class.items():
            print(f" - {class_name}: {', '.join(feats)}")
        if low_support_warnings:
            print("Data quality warnings:")
            for warning in low_support_warnings:
                print(f" - {warning}")
        print("Confusion matrix:")
        print(np.array(cm))

        return evaluation


def evaluate_random_forest_cross_run(
    train_csv_paths: Sequence[Path],
    test_csv_paths: Sequence[Path],
    rf_n_estimators: int = 200,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train a Random Forest on the union of `train_csv_paths` and evaluate it on
    the union of `test_csv_paths`. This is the honest cross-run evaluation
    described in the README: train on one batch of simulation runs, test on
    a held-out batch, so the test set never shares an attack episode with
    the training set.
    """
    if not train_csv_paths or not test_csv_paths:
        raise ValueError("Need at least one train CSV and one test CSV.")

    def _load_many(paths: Sequence[Path]) -> pd.DataFrame:
        frames = [pd.read_csv(Path(p)) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        required = {"timestamp", "bus", "arb_id", "payload", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Combined log missing columns: {sorted(missing)}")
        return df

    train_df = _load_many(train_csv_paths)
    test_df = _load_many(test_csv_paths)

    train_features = MLIDS._extract_features_from_df(train_df)
    test_features = MLIDS._extract_features_from_df(test_df)

    X_train = train_features[FEATURE_COLUMNS].fillna(0.0)
    y_train = train_features["label"].astype(int)
    X_test = test_features[FEATURE_COLUMNS].fillna(0.0)
    y_test = test_features["label"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Training set must contain at least two classes for cross-run RF.")

    clf = RandomForestClassifier(n_estimators=int(rf_n_estimators), random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    labels_sorted = sorted(set(y_train.unique().tolist()) | set(y_test.unique().tolist()))
    target_names = [ATTACK_NAME_BY_LABEL.get(i, str(i)) for i in labels_sorted]
    report = classification_report(
        y_test,
        y_pred,
        labels=labels_sorted,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted).tolist()

    proba_classes = [ATTACK_NAME_BY_LABEL.get(c, str(c)) for c in clf.classes_]
    conf_true_class = {}
    for class_name in ["injection", "fuzzing", "dos"]:
        if class_name in proba_classes:
            idx = proba_classes.index(class_name)
            class_label = clf.classes_[idx]
            mask = y_test.to_numpy() == class_label
            conf_true_class[class_name] = (
                float(np.mean(y_prob[mask, idx])) if np.any(mask) else 0.0
            )
        else:
            conf_true_class[class_name] = 0.0

    class_metrics = {
        ATTACK_NAME_BY_LABEL.get(lbl, str(lbl)): {
            "precision": float(report[ATTACK_NAME_BY_LABEL.get(lbl, str(lbl))]["precision"]),
            "recall": float(report[ATTACK_NAME_BY_LABEL.get(lbl, str(lbl))]["recall"]),
            "f1_score": float(report[ATTACK_NAME_BY_LABEL.get(lbl, str(lbl))]["f1-score"]),
        }
        for lbl in labels_sorted
    }

    test_support = {
        ATTACK_NAME_BY_LABEL.get(int(lbl), str(int(lbl))): int(count)
        for lbl, count in y_test.value_counts().to_dict().items()
    }

    importances = clf.feature_importances_
    top3 = [FEATURE_COLUMNS[i] for i in np.argsort(importances)[::-1][:3]]
    top_features_per_class = {
        ATTACK_NAME_BY_LABEL.get(label, str(label)): top3
        for label in labels_sorted
        if label != 0
    }

    result = {
        "mode": "cross_run",
        "train_runs": [str(p) for p in train_csv_paths],
        "test_runs": [str(p) for p in test_csv_paths],
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "overall_accuracy": acc,
        "per_class_metrics": class_metrics,
        "test_support": test_support,
        "rf_confidence_on_true_class": conf_true_class,
        "confusion_matrix": {"labels": target_names, "matrix": cm},
        "top_features_per_attack_class": top_features_per_class,
    }

    if output_path is not None:
        op = Path(output_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def _payload_to_var_mean(payload: Any) -> Tuple[float, float]:
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                values = [int(v) for v in text.strip("[]").split(",") if v.strip()]
            except ValueError:
                values = []
        else:
            clean = text.replace(" ", "")
            if clean.startswith("0x"):
                clean = clean[2:]
            try:
                values = list(bytes.fromhex(clean)) if clean else []
            except ValueError:
                values = []
    elif isinstance(payload, (bytes, bytearray)):
        values = list(payload)
    elif isinstance(payload, Iterable):
        try:
            values = [int(v) for v in payload]
        except Exception:
            values = []
    else:
        values = []

    values = (values[:8] + [0] * 8)[:8]
    arr = np.array(values, dtype=float)
    return float(arr.var()), float(arr.mean())


def _rolling_count_1s(timestamps: pd.Series) -> pd.Series:
    arr = timestamps.to_numpy(dtype=float)
    counts = np.zeros_like(arr, dtype=float)
    left = 0
    for right in range(arr.shape[0]):
        while arr[right] - arr[left] > 1.0:
            left += 1
        counts[right] = float(right - left + 1)
    return pd.Series(counts, index=timestamps.index)
