"""
Shared data structures used across all modules.
No logic here — only dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    frame_id: int


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    frame_id: int
    centroid_history: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ViolationRecord:
    violation_type: str
    confidence: float
    vehicle_id: int
    bbox: tuple[int, int, int, int]
    timestamp: str
    frame_id: int
    plate_number: Optional[str] = None
    plate_confidence: Optional[float] = None
    status: str = "pending"           # "auto_flagged" | "review" | "indeterminate"
    evidence_image_path: Optional[str] = None
    evidence_json_path: Optional[str] = None
    is_blurry: bool = False
    camera_id: str = "cam_001"
    # Region to run license-plate detection/OCR on. For most violations this is
    # the vehicle bbox itself. For helmet (where `bbox` is the tight head box),
    # this holds the full motorcycle box so ANPR can actually find a plate.
    anpr_bbox: Optional[tuple[int, int, int, int]] = None
    # SHA-256 of the saved evidence image, written for tamper-evidence / chain of custody.
    evidence_sha256: Optional[str] = None
