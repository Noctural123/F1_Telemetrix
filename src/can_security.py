from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CANMessage:
    timestamp: float
    bus: str
    can_id: int
    source: str
    target_driver: str
    payload: bytes


def _encode_u16(value: float, scale: float = 1.0) -> bytes:
    clamped = max(0, min(int(round(value * scale)), 65535))
    return clamped.to_bytes(2, byteorder="big", signed=False)


def _build_baseline_messages(timestamp: float, code: str, pos: Dict) -> List[CANMessage]:
    speed = float(pos.get("speed", 0.0))
    throttle = float(pos.get("throttle", 0.0))
    brake = float(pos.get("brake", 0.0))
    gear = int(pos.get("gear", 0))

    return [
        CANMessage(timestamp, "sensor_bus", 0x100, "sensor_ecu", code, _encode_u16(speed, 10.0)),
        CANMessage(timestamp, "sensor_bus", 0x101, "sensor_ecu", code, _encode_u16(throttle, 10.0)),
        CANMessage(timestamp, "powertrain_bus", 0x102, "powertrain_ecu", code, _encode_u16(brake, 10.0)),
        CANMessage(timestamp, "display_bus", 0x103, "display_ecu", code, bytes([max(0, min(gear, 15))])),
    ]


def _injection_messages(timestamp: float, target_driver: str) -> List[CANMessage]:
    forged_speed_kph = 380.0
    return [
        CANMessage(
            timestamp,
            "sensor_bus",
            0x100,
            "obd_gateway_attacker",
            target_driver,
            _encode_u16(forged_speed_kph, 10.0),
        )
    ]


def _fuzzing_messages(timestamp: float, target_driver: str, rng: random.Random, count: int = 8) -> List[CANMessage]:
    messages: List[CANMessage] = []
    for _ in range(count):
        fuzz_id = rng.randint(0x200, 0x7FF)
        payload_len = rng.randint(2, 8)
        payload = bytes(rng.randint(0, 255) for _ in range(payload_len))
        messages.append(
            CANMessage(
                timestamp,
                "powertrain_bus",
                fuzz_id,
                "obd_gateway_attacker",
                target_driver,
                payload,
            )
        )
    return messages


def build_security_overlay(
    frames: List[Dict],
    attack_type: str = "injection",
    target_driver: Optional[str] = None,
    attack_start_s: float = 20.0,
    attack_duration_s: float = 20.0,
    seed: int = 7,
) -> List[Dict]:
    """
    Build per-frame CAN traffic/attack metadata for UI rendering.
    This implements CAN baseline + attack simulation only (no IDS yet).
    """
    if not frames:
        return []

    first_frame_drivers = list(frames[0].get("drivers", {}).keys())
    if not first_frame_drivers:
        return []

    actual_target = target_driver if target_driver in first_frame_drivers else first_frame_drivers[0]
    rng = random.Random(seed)
    attack_end_s = attack_start_s + attack_duration_s
    overlay: List[Dict] = []

    for frame in frames:
        t = float(frame.get("t", 0.0))
        drivers = frame.get("drivers", {})
        all_messages: List[CANMessage] = []

        for code, pos in drivers.items():
            all_messages.extend(_build_baseline_messages(t, code, pos))

        attack_active = attack_start_s <= t <= attack_end_s
        attack_messages: List[CANMessage] = []
        if attack_active and actual_target in drivers:
            if attack_type == "fuzzing":
                attack_messages = _fuzzing_messages(t, actual_target, rng)
            else:
                attack_messages = _injection_messages(t, actual_target)
            all_messages.extend(attack_messages)

        bus_totals = {"sensor_bus": 0, "display_bus": 0, "powertrain_bus": 0}
        for msg in all_messages:
            bus_totals[msg.bus] = bus_totals.get(msg.bus, 0) + 1

        dominant_attack_id = attack_messages[0].can_id if attack_messages else None
        overlay.append(
            {
                "attack_active": attack_active,
                "attack_type": attack_type.upper(),
                "target_driver": actual_target,
                "total_messages": len(all_messages),
                "attack_messages": len(attack_messages),
                "dominant_attack_id": dominant_attack_id,
                "bus_totals": bus_totals,
            }
        )

    return overlay
