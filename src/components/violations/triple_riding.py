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

from datetime import datetime, timezone
from src.models import TrackedObject, ViolationRecord
from src.components.violations.classifier import route


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
        riders = [p for p in persons if _is_rider(moto.bbox, p.bbox, min_overlap_ratio)]
        if len(riders) >= 3:
            # Confidence is gated by the weakest underlying detection: a triple
            # built from low-confidence person boxes should not auto-flag.
            min_det_conf = min([moto.confidence] + [p.confidence for p in riders])
            record = ViolationRecord(
                violation_type="triple_riding",
                confidence=round(float(min_det_conf), 3),
                vehicle_id=moto.track_id,
                bbox=moto.bbox,
                timestamp=datetime.now(timezone.utc).isoformat(),
                frame_id=frame_id,
                camera_id=camera_id,
            )
            violations.append(route(record))
    return violations


def _is_rider(moto_box: tuple, person_box: tuple, min_overlap_ratio: float) -> bool:
    """
    A person counts as a rider of this motorcycle only if:
      1. Their box is substantially contained in the motorcycle box, AND
      2. Their horizontal centre lies within the motorcycle's x-span.
    The second test rejects pedestrians and riders of an adjacent bike that
    merely overlap the box from the side in 2D image space.
    """
    if _containment(moto_box, person_box) < min_overlap_ratio:
        return False
    ax1, _, ax2, _ = moto_box
    bx1, _, bx2, _ = person_box
    pcx = (bx1 + bx2) / 2
    return ax1 <= pcx <= ax2


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
