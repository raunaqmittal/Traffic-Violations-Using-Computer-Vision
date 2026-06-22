"""
Seatbelt non-compliance detector.

Crops the windshield ROI from a car bounding box and runs the pretrained
RISEF/yolov11s-seatbelt YOLOv11s classifier from HuggingFace.

Model: https://huggingface.co/RISEF/yolov11s-seatbelt
Classes: ["no_seatbelt", "seat_belt"]
License: AGPL-3.0 (same as Ultralytics)

On first run the weights (~19 MB) are downloaded via huggingface_hub and
cached at the path configured in pipeline.yaml (models.seatbelt_classifier).
Subsequent runs are fully offline.

Vehicle filtering:
  Only "car", "truck", and "bus" are checked. "person" is explicitly excluded
  because a pedestrian detected on the road is NOT inside a vehicle. Checking
  seatbelt compliance on a person standing outside a car would always fire a
  false positive — you cannot wear a seatbelt while standing on the road.
  The crop ROI (15%-55% of the bounding box height) targets the windshield/
  torso region of the *vehicle*, not of a standalone person.

If the crop is too small, or the download fails, the record is marked
"indeterminate" and routed to the human review queue.

Track memory integration: results are cached per car track. A car that was
already classified does not get re-checked until SEATBELT_REFRESH_INTERVAL
frames later. An indeterminate result is also cached so we don't flood the
DB with one indeterminate record per car per frame.
"""


import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.models import TrackedObject, ViolationRecord
from src.violations.classifier import route

log = logging.getLogger(__name__)

# HuggingFace repo for the pretrained seatbelt classifier.
_HF_REPO_ID = "RISEF/yolov11s-seatbelt"
_HF_FILENAME = "weights/best.pt"

# The YOLO classify model outputs these class names.
# Map them to the internal labels used by the rest of the pipeline.
_CLASS_TO_LABEL = {
    "no_seatbelt": "no_seatbelt",
    "seat_belt":   "seatbelt",
    # guard against minor name variations
    "seatbelt":    "seatbelt",
    "noseatbelt":  "no_seatbelt",
}


def _download_hf_model(cache_path: str) -> str | None:
    """
    Download RISEF/yolov11s-seatbelt from HuggingFace Hub to *cache_path*.
    Returns the resolved local path on success, or None on failure.
    """
    dest = Path(cache_path)
    if dest.exists():
        log.info("Seatbelt model already cached at %s", dest)
        return str(dest)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.error(
            "huggingface_hub is not installed. "
            "Run: pip install huggingface_hub>=0.23.0"
        )
        return None

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "Downloading seatbelt model from HuggingFace (%s / %s) → %s",
            _HF_REPO_ID, _HF_FILENAME, dest,
        )
        tmp_path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=_HF_FILENAME,
        )
        # Copy to the configured cache location so the pipeline can find it later.
        import shutil
        shutil.copy2(tmp_path, dest)
        log.info("Seatbelt model saved to %s", dest)
        return str(dest)
    except Exception as exc:
        log.error("Failed to download seatbelt model: %s", exc)
        return None


def _load_yolo_model(model_path: str, device: str):
    """Load the YOLO classify model. Returns None on failure."""
    try:
        import torch
        from ultralytics import YOLO
        
        model = YOLO(model_path, task="classify")
        # Warm up device assignment — ultralytics handles device internally
        # via predict() kwargs; store device for later use.
        model._seatbelt_device = device
        log.info("Seatbelt YOLO model loaded from %s (device=%s)", model_path, device)
        return model
    except Exception as exc:
        log.error("Failed to load seatbelt YOLO model: %s", exc)
        return None


class SeatbeltChecker:
    def __init__(
        self,
        model_path: str | None,
        conf_threshold: float = 0.55,
        device: str = "cpu",
        windshield_top_fraction: float = 0.15,
        windshield_bottom_fraction: float = 0.55,
        min_crop_width: int = 60,
        min_crop_height: int = 40,
    ):
        self.conf = conf_threshold
        self.device = device
        self.top_frac = windshield_top_fraction
        self.bot_frac = windshield_bottom_fraction
        self.min_w = min_crop_width
        self.min_h = min_crop_height
        self.model = self._init_model(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        tracks: list[TrackedObject],
        frame: np.ndarray,
        frame_id: int,
        camera_id: str = "cam_001",
        track_memory=None,
    ) -> list[ViolationRecord]:
        violations: list[ViolationRecord] = []

        # Only check enclosed vehicles: car, truck, bus.
        # "person" is deliberately excluded — a person detected by the vehicle
        # detector on the road is a pedestrian, not a car occupant. Running the
        # seatbelt model on a pedestrian's torso crop would always produce a
        # false positive (they have no seatbelt to wear outside a vehicle).
        cars = [t for t in tracks if t.class_name in ("car", "truck", "bus")]

        min_confirm = getattr(track_memory, "min_seatbelt_confirm", 1) if track_memory else 1

        for car in cars:
            # Skip if already cached and not yet due for recheck.
            if track_memory is not None and not track_memory.needs_seatbelt_check(car.track_id, frame_id):
                state = track_memory.get(car.track_id)
                # Emit a violation only once per track, only after min_confirm cycles agree.
                if (state
                        and state.seatbelt_status == "no_seatbelt"
                        and not state.seatbelt_violation_emitted
                        and state.seatbelt_confirm_count >= min_confirm):
                    record = ViolationRecord(
                        violation_type="seatbelt",
                        confidence=state.seatbelt_confidence,
                        vehicle_id=car.track_id,
                        bbox=car.bbox,
                        timestamp=datetime.utcnow().isoformat(),
                        frame_id=frame_id,
                        camera_id=camera_id,
                    )
                    violations.append(route(record))
                    state.seatbelt_violation_emitted = True
                continue

            crop = self._crop_windshield(frame, car.bbox)
            if crop is None or self.model is None:
                if track_memory is not None:
                    state = track_memory.get_or_create(car.track_id, "car")
                    if not state.seatbelt_checked:
                        state.seatbelt_checked = True
                        state.seatbelt_status = "indeterminate"
                        state.last_seatbelt_frame = frame_id
                        violations.append(self._indeterminate(car, frame_id, camera_id))
                else:
                    violations.append(self._indeterminate(car, frame_id, camera_id))
                continue

            confidence, label = self._classify(crop)

            if track_memory is not None:
                state = track_memory.get_or_create(car.track_id, "car")
                state.seatbelt_checked = True
                state.seatbelt_status = label
                state.seatbelt_confidence = confidence
                state.last_seatbelt_frame = frame_id

                if label == "no_seatbelt":
                    # Increment confirm counter on each recheck cycle seeing no_seatbelt.
                    state.seatbelt_confirm_count += 1
                else:
                    # Reset confirm counter when a clean (seatbelt visible) check is made.
                    state.seatbelt_confirm_count = 0

                # Emit only after enough confirm cycles.
                if (label == "no_seatbelt"
                        and not state.seatbelt_violation_emitted
                        and state.seatbelt_confirm_count >= min_confirm):
                    record = ViolationRecord(
                        violation_type="seatbelt",
                        confidence=confidence,
                        vehicle_id=car.track_id,
                        bbox=car.bbox,
                        timestamp=datetime.utcnow().isoformat(),
                        frame_id=frame_id,
                        camera_id=camera_id,
                    )
                    violations.append(route(record))
                    state.seatbelt_violation_emitted = True
            else:
                # No TrackMemory — emit immediately (single-frame mode, e.g. cloud demo).
                if label == "no_seatbelt":
                    record = ViolationRecord(
                        violation_type="seatbelt",
                        confidence=confidence,
                        vehicle_id=car.track_id,
                        bbox=car.bbox,
                        timestamp=datetime.utcnow().isoformat(),
                        frame_id=frame_id,
                        camera_id=camera_id,
                    )
                    violations.append(route(record))
        return violations

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_model(self, model_path: str | None):
        """
        Resolve model path:
        - If model_path is given and the file exists → load directly.
        - If model_path is given but file is missing → treat as the HF cache
          destination and auto-download.
        - If model_path is None → use default HF cache path and auto-download.
        """
        cache_path = model_path or os.path.join("models", "weights", "seatbelt_yolov11s.pt")
        resolved = _download_hf_model(cache_path)
        if resolved is None:
            log.warning(
                "Seatbelt model unavailable — all results will be 'indeterminate'."
            )
            return None
        return _load_yolo_model(resolved, self.device)

    def _crop_windshield(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        cy1 = y1 + int(h * self.top_frac)
        cy2 = y1 + int(h * self.bot_frac)
        crop = frame[cy1:cy2, x1:x2]
        if crop.shape[0] < self.min_h or crop.shape[1] < self.min_w:
            return None
        return crop

    def _classify(self, crop: np.ndarray) -> tuple[float, str]:
        """
        Run YOLO classify inference on the windshield crop.
        Returns (confidence, internal_label) where internal_label is one of
        "seatbelt" | "no_seatbelt".
        """
        # Convert BGR → RGB for YOLO
        pil_img = Image.fromarray(crop[:, :, ::-1])

        results = self.model.predict(
            source=pil_img,
            device=self.device,
            verbose=False,
        )

        probs = results[0].probs          # ultralytics Probs object
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        raw_name = self.model.names[top1_idx]  # e.g. "no_seatbelt" or "seat_belt"

        internal_label = _CLASS_TO_LABEL.get(raw_name.lower(), "seatbelt")

        # Apply local conf_threshold: if the model is uncertain, fall back to seatbelt
        if top1_conf < self.conf:
            return top1_conf, "seatbelt"

        return top1_conf, internal_label

    def _indeterminate(self, car: TrackedObject, frame_id: int, camera_id: str) -> ViolationRecord:
        return ViolationRecord(
            violation_type="seatbelt",
            confidence=0.0,
            vehicle_id=car.track_id,
            bbox=car.bbox,
            timestamp=datetime.utcnow().isoformat(),
            frame_id=frame_id,
            status="indeterminate",
            camera_id=camera_id,
        )
