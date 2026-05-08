from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


WHITELIST_ARB_IDS: frozenset = frozenset({0x100, 0x101, 0x102, 0x103})

# (signal_name, low, high) for the decoded payload value
PAYLOAD_BOUNDS: Dict[int, Tuple[str, float, float]] = {
    0x100: ("speed_kph", 0.0, 360.0),
    0x101: ("throttle_pct", 0.0, 100.0),
    0x102: ("brake_pct", 0.0, 100.0),
    0x103: ("gear", 0.0, 8.0),
}

VIOLATION_WHITELIST = "whitelist"
VIOLATION_FREQUENCY = "frequency"
VIOLATION_PAYLOAD = "payload_range"


@dataclass
class RuleBasedDetection:
    timestamp: float
    bus: str
    arb_id: int
    label: int
    violations: List[str] = field(default_factory=list)
    decoded_value: Optional[float] = None
    observed_rate_hz: float = 0.0
    baseline_rate_hz: float = 0.0


class RuleBasedIDS:
    """
    Streaming rule-based IDS with three independent checks:

    1. whitelist     -- arb_id must be in the configured allow-list.
    2. payload_range -- the decoded signal value must fall inside its
                        physical bounds (e.g. speed in [0, 360] km/h).
    3. frequency     -- the per-arb_id 1-second message rate must not
                        exceed `frequency_multiplier * baseline_rate`.
                        Arb-ids never seen in baseline are also flagged
                        once they exceed `unseen_id_min_burst` in 1s.

    The detector first observes traffic for `baseline_seconds` of
    *frame timestamp time* (not wall-clock) to learn per-arb_id baseline
    rates, then evaluates every subsequent frame against all three rules.
    """

    def __init__(
        self,
        baseline_seconds: float = 10.0,
        frequency_multiplier: float = 2.0,
        whitelist_arb_ids: Optional[frozenset] = None,
        payload_bounds: Optional[Dict[int, Tuple[str, float, float]]] = None,
        unseen_id_min_burst: int = 5,
    ) -> None:
        self.baseline_seconds = float(baseline_seconds)
        self.frequency_multiplier = float(frequency_multiplier)
        self.whitelist_arb_ids = whitelist_arb_ids or WHITELIST_ARB_IDS
        self.payload_bounds = payload_bounds or PAYLOAD_BOUNDS
        self.unseen_id_min_burst = int(unseen_id_min_burst)

        self._first_ts: Optional[float] = None
        self._baseline_finalized: bool = False
        self._baseline_counts: Dict[int, int] = defaultdict(int)
        self._baseline_rate_hz: Dict[int, float] = {}
        self._recent_ts: Dict[int, Deque[float]] = defaultdict(deque)

        self._detection_log: List[RuleBasedDetection] = []
        self._detections_by_label: Dict[int, int] = defaultdict(int)
        self._violation_counts: Dict[str, int] = defaultdict(int)
        self._frames_seen_by_label: Dict[int, int] = defaultdict(int)

    @staticmethod
    def _decode_value(arb_id: int, payload: bytes) -> Optional[float]:
        if arb_id in (0x100, 0x101, 0x102):
            if len(payload) < 2:
                return None
            raw = int.from_bytes(bytes(payload[:2]), byteorder="big", signed=False)
            return raw / 10.0
        if arb_id == 0x103:
            if len(payload) < 1:
                return None
            return float(payload[0])
        return None

    def _check_payload(self, arb_id: int, payload: bytes) -> Tuple[bool, Optional[float]]:
        bounds = self.payload_bounds.get(arb_id)
        if bounds is None:
            # Unknown arb_id: payload range cannot be checked here, but the
            # whitelist rule will catch it instead.
            return True, None
        _, lo, hi = bounds
        value = self._decode_value(arb_id, payload)
        if value is None:
            return False, None
        return lo <= value <= hi, value

    def _finalize_baseline(self, current_ts: float) -> None:
        if self._baseline_finalized or self._first_ts is None:
            return
        elapsed = max(current_ts - self._first_ts, 1e-6)
        self._baseline_rate_hz = {
            arb_id: count / elapsed for arb_id, count in self._baseline_counts.items()
        }
        self._baseline_finalized = True

    def process_frame(
        self,
        timestamp: float,
        bus: str,
        arb_id: int,
        payload: Any,
        label: int = 0,
    ) -> Optional[RuleBasedDetection]:
        ts = float(timestamp)
        if isinstance(payload, (bytes, bytearray)):
            payload_bytes = bytes(payload)
        elif isinstance(payload, str):
            try:
                payload_bytes = bytes.fromhex(payload)
            except ValueError:
                payload_bytes = b""
        else:
            try:
                payload_bytes = bytes(int(b) & 0xFF for b in payload)
            except Exception:
                payload_bytes = b""

        if self._first_ts is None:
            self._first_ts = ts
        self._frames_seen_by_label[int(label)] += 1

        bucket = self._recent_ts[arb_id]
        bucket.append(ts)
        while bucket and (ts - bucket[0]) > 1.0:
            bucket.popleft()

        in_baseline = (ts - self._first_ts) < self.baseline_seconds
        if in_baseline:
            self._baseline_counts[arb_id] += 1
            return None

        if not self._baseline_finalized:
            self._finalize_baseline(ts)

        violations: List[str] = []

        if arb_id not in self.whitelist_arb_ids:
            violations.append(VIOLATION_WHITELIST)

        payload_ok, decoded = self._check_payload(arb_id, payload_bytes)
        if not payload_ok:
            violations.append(VIOLATION_PAYLOAD)

        observed_rate = float(len(bucket))
        baseline_rate = self._baseline_rate_hz.get(arb_id)
        if baseline_rate is not None and baseline_rate > 0.0:
            if observed_rate > self.frequency_multiplier * baseline_rate:
                violations.append(VIOLATION_FREQUENCY)
        elif baseline_rate is None and observed_rate >= self.unseen_id_min_burst:
            # Arb-id never observed during baseline, now bursting.
            violations.append(VIOLATION_FREQUENCY)

        if not violations:
            return None

        for v in violations:
            self._violation_counts[v] += 1
        self._detections_by_label[int(label)] += 1
        det = RuleBasedDetection(
            timestamp=ts,
            bus=str(bus),
            arb_id=int(arb_id),
            label=int(label),
            violations=violations,
            decoded_value=decoded,
            observed_rate_hz=observed_rate,
            baseline_rate_hz=float(baseline_rate) if baseline_rate is not None else 0.0,
        )
        self._detection_log.append(det)
        return det

    def summary(self) -> Dict[str, Any]:
        return {
            "baseline_seconds": self.baseline_seconds,
            "frequency_multiplier": self.frequency_multiplier,
            "baseline_finalized": self._baseline_finalized,
            "baseline_rate_hz": {int(k): float(v) for k, v in self._baseline_rate_hz.items()},
            "total_detections": int(sum(self._detections_by_label.values())),
            "violations_by_type": {str(k): int(v) for k, v in self._violation_counts.items()},
            "detections_by_label": {int(k): int(v) for k, v in self._detections_by_label.items()},
            "frames_seen_by_label": {int(k): int(v) for k, v in self._frames_seen_by_label.items()},
        }
