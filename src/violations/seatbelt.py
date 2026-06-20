"""
Seatbelt non-compliance detector.
Crops the windshield ROI from a car bounding box and runs a lightweight
binary CNN classifier (seatbelt / no_seatbelt).

If the crop is too small or the model is not loaded, the record is
marked "indeterminate" and routed to the human review queue.

Track memory integration: results are cached per car track. A car that was
already classified does not get re-checked until SEATBELT_REFRESH_INTERVAL
frames later. An indeterminate result is also cached so we don't flood the
DB with one indeterminate record per car per frame.
"""

import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from src.models import TrackedObject, ViolationRecord
from src.violations.classifier import route


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
        self.model = self._load_model(model_path) if model_path else None
        self._transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def check(
        self,
        tracks: list[TrackedObject],
        frame: np.ndarray,
        frame_id: int,
        camera_id: str = "cam_001",
        track_memory=None,
    ) -> list[ViolationRecord]:
        violations: list[ViolationRecord] = []
        cars = [t for t in tracks if t.class_name == "car"]

        for car in cars:
            # Skip if already cached and not yet due for recheck.
            if track_memory is not None and not track_memory.needs_seatbelt_check(car.track_id, frame_id):
                state = track_memory.get(car.track_id)
                # Emit a violation only once per track.
                if state and state.seatbelt_status == "no_seatbelt" and not state.seatbelt_violation_emitted:
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
                if track_memory is not None:
                    state.seatbelt_violation_emitted = True
        return violations

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
        pil_img = Image.fromarray(crop[:, :, ::-1])
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits).item()
        if prob >= self.conf:
            return prob, "no_seatbelt"
        return 1.0 - prob, "seatbelt"

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

    def _load_model(self, path: str) -> nn.Module | None:
        try:
            model = _SeatbeltCNN()
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            return model.to(self.device)
        except Exception:
            return None


class _SeatbeltCNN(nn.Module):
    """Lightweight binary CNN for windshield crop classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Linear(64 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
