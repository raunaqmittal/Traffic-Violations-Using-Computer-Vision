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

Track memory integration: the helmet model is only run when at least one
motorcycle needs a (re-)check, as decided by TrackMemory. Results are cached
so a bike that was checked 2 frames ago does not trigger another YOLO call.
"""

import numpy as np
from datetime import datetime
from ultralytics import YOLO
from src.models import TrackedObject, ViolationRecord
from src.violations.classifier import route

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
        self.head_roi_fraction = head_roi_fraction
        self.flag_invalid_helmet = flag_invalid_helmet

    def check(
        self,
        tracks: list[TrackedObject],
        frame: np.ndarray,
        frame_id: int,
        camera_id: str = "cam_001",
        track_memory=None,
    ) -> list[ViolationRecord]:
        motorcycles = [t for t in tracks if t.class_name == "motorcycle"]
        if not motorcycles:
            return []

        # Determine which motorcycles actually need a fresh helmet check.
        if track_memory is not None:
            needs_check = [m for m in motorcycles if track_memory.needs_helmet_check(m.track_id, frame_id)]
        else:
            needs_check = motorcycles

        # If no motorcycle needs checking, replay cached violations for tracks
        # that had a violation result but haven't emitted it yet this cycle.
        if not needs_check:
            return self._replay_cached(motorcycles, frame_id, camera_id, track_memory)

        # Run the full-frame helmet model once.
        results = self.model.predict(source=frame, conf=self.conf, device=self.device, verbose=False)
        if not results or not results[0].boxes:
            # No helmet detections — mark all needing-check motorcycles as "ok".
            if track_memory is not None:
                for m in needs_check:
                    state = track_memory.get_or_create(m.track_id, "motorcycle")
                    state.helmet_checked = True
                    state.helmet_status = "ok"
                    state.helmet_confidence = 0.0
                    state.last_helmet_frame = frame_id
            return []

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

        violations: list[ViolationRecord] = []
        for moto in motorcycles:
            conf = self._associate(moto.bbox, violation_heads)

            if track_memory is not None:
                state = track_memory.get_or_create(moto.track_id, "motorcycle")
                if moto in needs_check:
                    state.helmet_checked = True
                    state.last_helmet_frame = frame_id
                    if conf is not None:
                        state.helmet_status = "no_helmet"
                        state.helmet_confidence = conf
                    else:
                        state.helmet_status = "ok"
                        state.helmet_confidence = 0.0

                # Emit violation only once per track per detection cycle.
                if state.helmet_status == "no_helmet" and not state.helmet_violation_emitted:
                    record = ViolationRecord(
                        violation_type="helmet",
                        confidence=state.helmet_confidence,
                        vehicle_id=moto.track_id,
                        bbox=moto.bbox,
                        timestamp=datetime.utcnow().isoformat(),
                        frame_id=frame_id,
                        camera_id=camera_id,
                    )
                    violations.append(route(record))
                    state.helmet_violation_emitted = True
            else:
                if conf is not None:
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

    def _replay_cached(
        self,
        motorcycles: list[TrackedObject],
        frame_id: int,
        camera_id: str,
        track_memory,
    ) -> list[ViolationRecord]:
        """Return violations from cache for tracks that haven't emitted yet."""
        if track_memory is None:
            return []
        violations = []
        for moto in motorcycles:
            state = track_memory.get(moto.track_id)
            if state and state.helmet_status == "no_helmet" and not state.helmet_violation_emitted:
                record = ViolationRecord(
                    violation_type="helmet",
                    confidence=state.helmet_confidence,
                    vehicle_id=moto.track_id,
                    bbox=moto.bbox,
                    timestamp=datetime.utcnow().isoformat(),
                    frame_id=frame_id,
                    camera_id=camera_id,
                )
                violations.append(route(record))
                state.helmet_violation_emitted = True
        return violations

    def _associate(
        self,
        moto_bbox: tuple[int, int, int, int],
        heads: list[tuple[tuple[int, int, int, int], float]],
    ) -> float | None:
        x1, y1, x2, y2 = moto_bbox
        w, h = x2 - x1, y2 - y1
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
