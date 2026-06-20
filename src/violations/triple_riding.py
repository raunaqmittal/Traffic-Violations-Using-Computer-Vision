"""
Triple riding violation detector.
Rule: count persons whose bounding box sits substantially inside a motorcycle
bounding box. If count >= 3, flag the motorcycle track as a triple-riding
violation.

We use *containment* (intersection over the person's own area), NOT IoU.
Riders are small relative to the whole motorcycle+rider box, so a person fully
on the bike scores a low IoU (~0.1) yet a high containment (~1.0). IoU would
miss real triple-riding cases; containment captures them correctly.
"""

from datetime import datetime
from src.models import TrackedObject, ViolationRecord
from src.violations.classifier import route


def check(
    tracks: list[TrackedObject],
    frame_id: int,
    camera_id: str = "cam_001",
    min_overlap_ratio: float = 0.5,
) -> list[ViolationRecord]:
    violations: list[ViolationRecord] = []

    motorcycles = [t for t in tracks if t.class_name == "motorcycle"]
    persons = [t for t in tracks if t.class_name == "person"]

    for moto in motorcycles:
        riders = [p for p in persons if _containment(moto.bbox, p.bbox) >= min_overlap_ratio]
        if len(riders) >= 3:
            record = ViolationRecord(
                violation_type="triple_riding",
                confidence=1.0,
                vehicle_id=moto.track_id,
                bbox=moto.bbox,
                timestamp=datetime.utcnow().isoformat(),
                frame_id=frame_id,
                camera_id=camera_id,
            )
            violations.append(route(record))
    return violations


def _containment(moto_box: tuple, person_box: tuple) -> float:
    """Fraction of the person box that lies inside the motorcycle box."""
    ax1, ay1, ax2, ay2 = moto_box
    bx1, by1, bx2, by2 = person_box
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    person_area = (bx2 - bx1) * (by2 - by1)
    return inter / (person_area + 1e-6)
