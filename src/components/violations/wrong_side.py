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
    allowed_direction_deg,
    direction_tolerance_deg: float,
    min_track_frames: int = 8,
    consecutive_wrong_frames: int = 5,
    camera_id: str = "cam_001",
) -> list[ViolationRecord]:
    violations: list[ViolationRecord] = []
    vehicle_classes = {"car", "truck", "bus", "motorcycle", "auto-rickshaw", "three-wheeler"}

    # A road can be two-way / multi-lane, so more than one direction may be
    # legal. Accept either a single angle or a list of allowed angles; a vehicle
    # is only flagged if it disagrees with EVERY allowed direction.
    if isinstance(allowed_direction_deg, (list, tuple)):
        allowed_dirs = [float(a) for a in allowed_direction_deg]
    else:
        allowed_dirs = [float(allowed_direction_deg)]

    for track in tracks:
        if track.class_name not in vehicle_classes:
            continue
        history = track.centroid_history
        if len(history) < min_track_frames:
            continue

        # Count consecutive wrong-direction frames at the tail of the history
        wrong_count = _count_consecutive_wrong(
            history,
            allowed_dirs,
            direction_tolerance_deg,
        )
        if wrong_count >= consecutive_wrong_frames:
            # More consecutive wrong-direction frames = stronger evidence;
            # also weighted by how confident the detector was in the vehicle.
            evidence = min(0.70 + 0.03 * wrong_count, 1.0)
            record = ViolationRecord(
                violation_type="wrong_side",
                confidence=round(evidence * float(track.confidence), 3),
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
    allowed_dirs: list[float],
    tolerance_deg: float,
) -> int:
    """Count how many consecutive tail-frames move against ALL allowed directions."""
    count = 0
    for i in range(len(history) - 1, 0, -1):
        dx = history[i][0] - history[i - 1][0]
        dy = history[i][1] - history[i - 1][1]
        if abs(dx) < 2 and abs(dy) < 2:
            # Stationary — skip
            continue
        # Image coords: y increases downward, so flip dy for standard angles
        angle = math.degrees(math.atan2(-dy, dx)) % 360
        # Within tolerance of ANY legal direction -> this frame is fine.
        matches_any = False
        for allowed_deg in allowed_dirs:
            diff = abs(angle - allowed_deg) % 360
            diff = min(diff, 360 - diff)
            if diff <= tolerance_deg:
                matches_any = True
                break
        if not matches_any:
            count += 1
        else:
            break
    return count
