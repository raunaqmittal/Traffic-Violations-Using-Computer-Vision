"""
Illegal parking violation detector.
Rule: vehicle centroid inside a no-parking polygon for >= dwell_time_seconds seconds.

Dwell time is tracked per track ID using timestamps of first entry into the zone.
"""

from datetime import datetime, timezone
import numpy as np
from src.models import TrackedObject, ViolationRecord
from src.components.violations.classifier import route


# State is keyed by (camera_id, track_id) so that track IDs from different
# cameras never collide in a multi-camera deployment.
# dwell_tracker[(camera_id, track_id)] = (zone_name, entry_frame)
_dwell_tracker: dict[tuple[str, int], tuple[str, int]] = {}
_already_flagged: set[tuple[str, int]] = set()


def check(
    tracks: list[TrackedObject],
    frame_id: int,
    no_parking_zones: list[dict],
    fps: float = 25.0,
    dwell_time_seconds: float = 180.0,
    stationary_pixel_threshold: int = 15,
    camera_id: str = "cam_001",
) -> list[ViolationRecord]:
    violations: list[ViolationRecord] = []
    
    # Calculate how many frames correspond to the desired dwell time
    dwell_frames = int(dwell_time_seconds * fps)

    vehicle_classes = {"car", "truck", "bus", "auto-rickshaw", "three-wheeler"}

    for track in tracks:
        if track.class_name not in vehicle_classes:
            continue
        key = (camera_id, track.track_id)
        if key in _already_flagged:
            continue

        cx, cy = _centroid(track.bbox)

        zone_match = None
        for zone in no_parking_zones:
            polygon = np.array(zone["polygon"], dtype=np.int32)
            if _point_in_polygon(cx, cy, polygon):
                zone_match = zone
                break

        if zone_match is None:
            # Outside all zones — reset dwell timer
            _dwell_tracker.pop(key, None)
            continue

        if not _is_stationary(track.centroid_history, stationary_pixel_threshold):
            # Moving through zone — reset timer
            _dwell_tracker.pop(key, None)
            continue

        if key not in _dwell_tracker:
            _dwell_tracker[key] = (zone_match["name"], frame_id)
            continue

        zone_name, entry_frame = _dwell_tracker[key]
        if frame_id - entry_frame >= dwell_frames:
            _already_flagged.add(key)
            record = ViolationRecord(
                violation_type="illegal_parking",
                confidence=1.0,
                vehicle_id=track.track_id,
                bbox=track.bbox,
                timestamp=datetime.now(timezone.utc).isoformat(),
                frame_id=frame_id,
                camera_id=camera_id,
            )
            violations.append(route(record))
    return violations


def reset():
    _dwell_tracker.clear()
    _already_flagged.clear()


def _centroid(bbox: tuple) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def _is_stationary(history: list[tuple[int, int]], threshold: int) -> bool:
    if len(history) < 5:
        return False
    recent = history[-5:]
    total_motion = sum(
        abs(recent[i][0] - recent[i - 1][0]) + abs(recent[i][1] - recent[i - 1][1])
        for i in range(1, len(recent))
    )
    return total_motion / len(recent) < threshold


def _point_in_polygon(x: int, y: int, polygon: np.ndarray) -> bool:
    result = cv2_point_poly_test(x, y, polygon)
    return result


def cv2_point_poly_test(x: int, y: int, polygon: np.ndarray) -> bool:
    import cv2
    return cv2.pointPolygonTest(polygon.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0
