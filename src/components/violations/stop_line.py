"""
Stop-line violation detector.
Rule: vehicle centroid crosses the virtual stop line when signal state is "red" or "unknown".

Stop line is defined as two [x, y] points in cameras.yaml.
A vehicle "crosses" if its centroid is past the line (toward the intersection)
by at least crossing_margin_px pixels.
"""

import numpy as np
from datetime import datetime, timezone
from src.models import TrackedObject, ViolationRecord
from src.components.violations.classifier import route
from src.components.violations.signal_utils import detect_signal_state


_ALREADY_FLAGGED: set[int] = set()


def check(
    tracks: list[TrackedObject],
    frame: np.ndarray,
    frame_id: int,
    stop_line: list[list[int]],
    signal_roi: tuple[int, int, int, int],
    crossing_margin_px: int = 5,
    red_pixel_fraction: float = 0.15,
    green_pixel_fraction: float = 0.10,
    camera_id: str = "cam_001",
) -> list[ViolationRecord]:
    violations: list[ViolationRecord] = []

    signal = detect_signal_state(frame, signal_roi, red_pixel_fraction, green_pixel_fraction)
    if signal == "green":
        return violations

    vehicle_classes = {"car", "truck", "bus", "motorcycle", "auto-rickshaw", "three-wheeler"}
    p1 = np.array(stop_line[0], dtype=float)
    p2 = np.array(stop_line[1], dtype=float)

    for track in tracks:
        if track.class_name not in vehicle_classes:
            continue
        if track.track_id in _ALREADY_FLAGGED:
            continue

        cx, cy = _centroid(track.bbox)
        signed_dist = _signed_distance_to_line(p1, p2, (cx, cy))

        if signed_dist > crossing_margin_px:
            _ALREADY_FLAGGED.add(track.track_id)
            record = ViolationRecord(
                violation_type="stop_line",
                confidence=0.93 if signal == "red" else 0.75,
                vehicle_id=track.track_id,
                bbox=track.bbox,
                timestamp=datetime.now(timezone.utc).isoformat(),
                frame_id=frame_id,
                camera_id=camera_id,
            )
            violations.append(route(record))
    return violations


def reset_flagged():
    """Call this when processing a new video clip."""
    _ALREADY_FLAGGED.clear()


def _centroid(bbox: tuple) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def _signed_distance_to_line(p1: np.ndarray, p2: np.ndarray, point: tuple) -> float:
    """Positive distance means the point is on the 'past the line' side."""
    line = p2 - p1
    normal = np.array([-line[1], line[0]], dtype=float)
    normal /= (np.linalg.norm(normal) + 1e-6)
    vec = np.array(point, dtype=float) - p1
    return float(np.dot(vec, normal))
