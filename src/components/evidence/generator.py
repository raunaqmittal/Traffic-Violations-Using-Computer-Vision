"""
Evidence generator.
For each ViolationRecord, draws annotation on the frame,
saves a JPEG, writes a JSON sidecar, and updates the record with paths.
"""

import cv2
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from src.models import ViolationRecord


_VIOLATION_COLORS = {
    "helmet": (0, 0, 255),
    "seatbelt": (255, 100, 0),
    "triple_riding": (0, 165, 255),
    "wrong_side": (180, 0, 255),
    "stop_line": (0, 255, 255),
    "red_light": (0, 0, 200),
    "illegal_parking": (255, 0, 100),
}
_DEFAULT_COLOR = (0, 200, 0)


class EvidenceGenerator:
    def __init__(self, save_dir: str, jpeg_quality: int = 85):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.jpeg_quality = jpeg_quality

    def save(self, frame: np.ndarray, record: ViolationRecord) -> ViolationRecord:
        annotated = self._annotate(frame.copy(), record)
        stem = self._make_stem(record)
        img_path = self.save_dir / f"{stem}.jpg"
        json_path = self.save_dir / f"{stem}.json"

        cv2.imwrite(str(img_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        metadata = {
            "violation_type": record.violation_type,
            "confidence": record.confidence,
            "status": record.status,
            "vehicle_id": record.vehicle_id,
            "plate_number": record.plate_number,
            "plate_confidence": record.plate_confidence,
            "timestamp": record.timestamp,
            "frame_id": record.frame_id,
            "camera_id": record.camera_id,
            "bbox": list(record.bbox),
            "is_blurry": record.is_blurry,
            "evidence_image": str(img_path),
        }
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        record.evidence_image_path = str(img_path)
        record.evidence_json_path = str(json_path)
        return record

    def _annotate(self, frame: np.ndarray, record: ViolationRecord) -> np.ndarray:
        color = _VIOLATION_COLORS.get(record.violation_type, _DEFAULT_COLOR)
        x1, y1, x2, y2 = record.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{record.violation_type.upper()} {record.confidence:.2f}"
        if record.plate_number:
            label += f" | {record.plate_number}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        ts_label = f"ID:{record.vehicle_id} | {record.timestamp}"
        cv2.putText(frame, ts_label, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return frame

    def _make_stem(self, record: ViolationRecord) -> str:
        ts = record.timestamp.replace(":", "-").replace(".", "-")
        plate = record.plate_number.replace(" ", "_") if record.plate_number else "UNK"
        return f"{record.violation_type}_{record.vehicle_id}_{plate}_{ts}"
