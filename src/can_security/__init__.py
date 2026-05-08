from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .ml_ids import MLIDS, evaluate_random_forest_cross_run
from .rule_based_ids import RuleBasedIDS
from .run_simulation import run_full_ml_pipeline_from_frames, run_ml_evaluation_phase

_legacy_path = Path(__file__).resolve().parent.parent / "can_security.py"
_legacy_spec = importlib.util.spec_from_file_location("src._can_security_legacy", _legacy_path)
if _legacy_spec is None or _legacy_spec.loader is None:
    raise ImportError(f"Unable to load legacy CAN security module at {_legacy_path}")
_legacy_module = importlib.util.module_from_spec(_legacy_spec)
sys.modules[_legacy_spec.name] = _legacy_module
_legacy_spec.loader.exec_module(_legacy_module)

# Re-export existing API so prior imports keep working.
CANMessage = _legacy_module.CANMessage
build_security_overlay = _legacy_module.build_security_overlay

__all__ = [
    "CANMessage",
    "MLIDS",
    "RuleBasedIDS",
    "build_security_overlay",
    "evaluate_random_forest_cross_run",
    "run_ml_evaluation_phase",
    "run_full_ml_pipeline_from_frames",
]
