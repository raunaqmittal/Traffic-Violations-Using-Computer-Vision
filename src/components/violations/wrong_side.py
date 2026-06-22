"""
Wrong-side driving violation detector.
Rule: compute displacement vector of the track centroid over recent N frames.
If the direction deviates from the camera's allowed_direction_deg by more than
the tolerance for consecutive_wrong_frames frames, flag the vehicle.
"""

import math
from datetime import datetime, timezone
from src.models import TrackedObject, ViolationRecord
from src.components.violations.classifier import route


def check(
    tracks: list[TrackedObject],
    frame_id: int,
    allowed_direction_deg: float,
    direction_tolerance_deg: float,
    min_track_frames: int = 8,
    consecutive_wrong_frames: int = 5,
    camera_id: str = "cam_001",
) -> list[ViolationRecord]:
    violations: list[ViolationRecord] = []
    vehicle_classes = {"car", "truck", "bus", "motorcycle", "auto-rickshaw", "three-wheeler"}

    for track in tracks:
        if track.class_name not in vehicle_classes:
            continue
        history = track.centroid_history
        if len(history) < min_track_frames:
            continue

        # Count consecutive wrong-direction frames at the tail of the history
        wrong_count = _count_consecutive_wrong(
            history,
            allowed_direction_deg,
            direction_tolerance_deg,
        )
        if wrong_count >= consecutive_wrong_frames:
            record = ViolationRecord(
                violation_type="wrong_side",
                confidence=min(0.70 + 0.03 * wrong_count, 1.0),
                vehicle_id=track.track_id,
                bbox=track.bbox,
                timestamp=datetime.now(timezone.utc).isoformat(),
                frame_id=frame_id,
                camera_id=camera_id,
            )
            violations.append(route(record))
    return violations


def _count_consecutive_wrong(
    history: list[tuple[int, int]],
    allowed_deg: float,
    tolerance_deg: float,
) -> int:
    """Count how many consecutive tail-frames have wrong-direction motion."""
    count = 0
    for i in range(len(history) - 1, 0, -1):
        dx = history[i][0] - history[i - 1][0]
        dy = history[i][1] - history[i - 1][1]
        if abs(dx) < 2 and abs(dy) < 2:
            # Stationary — skip
            continue
        # Image coords: y increases downward, so flip dy for standard angles
        angle = math.degrees(math.atan2(-dy, dx)) % 360
        diff = abs(angle - allowed_deg) % 360
        diff = min(diff, 360 - diff)
        if diff > tolerance_deg:
            count += 1
        else:
            break
    return count
