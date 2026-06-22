"""
License plate localization using YOLOv8 fine-tuned on Indian plates.
Returns bounding boxes for detected plates within a given vehicle crop or full frame.
"""

import numpy as np
from ultralytics import YOLO
from src.models import Detection


class PlateDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.50, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.device = device

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            device=self.device,
            verbose=False,
            imgsz=1280,
        )
        plates: list[Detection] = []
        if not results:
            return plates

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plates.append(Detection(
                class_name="license_plate",
                confidence=float(box.conf[0]),
                bbox=(x1, y1, x2, y2),
                frame_id=frame_id,
            ))
        return plates

    def detect_in_vehicle_crop(
        self,
        frame: np.ndarray,
        vehicle_bbox: tuple[int, int, int, int],
        frame_id: int,
    ) -> list[Detection]:
        x1, y1, x2, y2 = vehicle_bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        plates = self.detect(crop, frame_id)
        # Translate crop-relative coords back to full-frame coords
        for p in plates:
            px1, py1, px2, py2 = p.bbox
            p.bbox = (px1 + x1, py1 + y1, px2 + x1, py2 + y1)
        return plates
