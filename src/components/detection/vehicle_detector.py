"""
Vehicle and road-user detector using YOLOv8.
Returns a list of Detection objects per frame.

Supported classes (COCO subset + fine-tuned extras):
  car, truck, bus, motorcycle, person, auto-rickshaw

Auto-rickshaw note: COCO does not include auto-rickshaws.
If the pretrained model misses them, fine-tune on IDD dataset
and update the model path in pipeline.yaml.
"""

import numpy as np
from ultralytics import YOLO
from src.models import Detection


VEHICLE_CLASSES = {
    "car", "truck", "bus", "motorcycle", "person",
    "auto-rickshaw", "three-wheeler",
}


class VehicleDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.45, nms_iou: float = 0.45, device: str = "cpu"):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = nms_iou
        self.device = device

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        for box in results[0].boxes:
            class_name = self.model.names[int(box.cls[0])]
            if class_name not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(Detection(
                class_name=class_name,
                confidence=float(box.conf[0]),
                bbox=(x1, y1, x2, y2),
                frame_id=frame_id,
            ))
        return detections
