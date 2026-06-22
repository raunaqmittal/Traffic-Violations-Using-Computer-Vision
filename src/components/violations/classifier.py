"""
Violation record classifier and confidence-based routing.

Every violation module calls route() to assign a status:
  - "auto_flagged"  : confidence >= threshold -> saved as confirmed evidence
  - "review"        : confidence < threshold  -> sent to human review queue
  - "indeterminate" : explicitly uncertain (e.g. seatbelt with bad crop)
"""

from src.models import ViolationRecord
from src.config import load_violations

_VIOLATION_CONFIG: dict = {}


def _get_threshold(violation_type: str) -> float:
    global _VIOLATION_CONFIG
    if not _VIOLATION_CONFIG:
        _VIOLATION_CONFIG = load_violations()
    vconf = _VIOLATION_CONFIG.get(violation_type, {})
    return float(vconf.get("auto_approve_confidence", 0.90))


def route(record: ViolationRecord) -> ViolationRecord:
    if record.status == "indeterminate":
        return record
    threshold = _get_threshold(record.violation_type)
    record.status = "auto_flagged" if record.confidence >= threshold else "review"
    return record
