"""
Helmet non-compliance detector.

The helmet model is a full-scene detector trained to localize riders and their
helmet state (rider_no_helmet, rider_full_face, ...). We therefore run it once
on the FULL frame and associate each `no_helmet` head detection to a motorcycle
track, rather than cropping a fixed top-fraction of the motorcycle box.

Why not crop the motorcycle box: COCO `motorcycle` boxes usually exclude the
rider's head (it sits above the bike box), so a top-fraction crop frequently
misses the head entirely. Full-frame detection + association uses the model the
way it was trained and is also cheaper when several motorcycles are present.
"""

import numpy as np
from datetime import datetime
from ultralytics import YOLO
from src.models import TrackedObject, ViolationRecord
from src.violations.classifier import route

# Substrings that mark a head detection as a helmet violation.
_NO_HELMET_MARKERS = ("no_helmet", "no-helmet", "without", "nohelmet")


class HelmetChecker:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.55,
        device: str = "cpu",
        head_roi_fraction: float = 0.35,
        flag_invalid_helmet: bool = False,
    ):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.device = device
        # Retained for config compatibility; used as the upward expansion factor
        # when associating a head detection to a motorcycle box.
        self.head_roi_fraction = head_roi_fraction
        # If True, an improperly-worn ("invalid") helmet also counts as a violation.
        self.flag_invalid_helmet = flag_invalid_helmet

    def check(
        self,
        tracks: list[TrackedObject],
        frame: np.ndarray,
        frame_id: int,
        camera_id: str = "cam_001",
    ) -> list[ViolationRecord]:
        motorcycles = [t for t in tracks if t.class_name == "motorcycle"]
        if not motorcycles:
            return []

        results = self.model.predict(source=frame, conf=self.conf, device=self.device, verbose=False)
        if not results or not results[0].boxes:
            return []

        # Collect head boxes that represent a violation.
        violation_heads: list[tuple[tuple[int, int, int, int], float]] = []
        for box in results[0].boxes:
            label = self.model.names[int(box.cls[0])].lower()
            is_violation = any(m in label for m in _NO_HELMET_MARKERS)
            if self.flag_invalid_helmet and "invalid" in label:
                is_violation = True
            if not is_violation:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            violation_heads.append(((x1, y1, x2, y2), float(box.conf[0])))

        if not violation_heads:
            return []

        violations: list[ViolationRecord] = []
        for moto in motorcycles:
            conf = self._associate(moto.bbox, violation_heads)
            if conf is None:
                continue
            record = ViolationRecord(
                violation_type="helmet",
                confidence=conf,
                vehicle_id=moto.track_id,
                bbox=moto.bbox,
                timestamp=datetime.utcnow().isoformat(),
                frame_id=frame_id,
                camera_id=camera_id,
            )
            violations.append(route(record))
        return violations

    def _associate(
        self,
        moto_bbox: tuple[int, int, int, int],
        heads: list[tuple[tuple[int, int, int, int], float]],
    ) -> float | None:
        """Return the best (max) confidence of a no-helmet head belonging to this
        motorcycle, or None. A head belongs to the bike if its centre falls inside
        the motorcycle box expanded upward (to include the rider's head) and
        slightly sideways."""
        x1, y1, x2, y2 = moto_bbox
        w, h = x2 - x1, y2 - y1
        # Expand: up by ~1 bike-height (head sits above the bike), small side margin.
        ex1 = x1 - 0.15 * w
        ex2 = x2 + 0.15 * w
        ey1 = y1 - 1.2 * h
        ey2 = y2 + 0.1 * h

        best_conf = None
        for (hx1, hy1, hx2, hy2), conf in heads:
            cx = (hx1 + hx2) / 2
            cy = (hy1 + hy2) / 2
            if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                if best_conf is None or conf > best_conf:
                    best_conf = conf
        return best_conf
