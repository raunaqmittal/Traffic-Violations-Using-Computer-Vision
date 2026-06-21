"""
Interactive demo for the Traffic Violation Detection System.

This is the public, CPU-friendly demo entry point (deploy target for
Streamlit Community Cloud / Hugging Face Spaces). Upload a traffic image and the
app runs the real detectors + violation logic and returns annotated evidence.

It reuses the exact production modules in src/ (VehicleDetector, HelmetChecker,
triple_riding, PlateDetector, PlateReader) so the demo behaviour matches the
full pipeline — it just runs per-image instead of per-video-frame.

Run locally:   streamlit run streamlit_app.py
Deploy:        see docs/DEPLOYMENT.md
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models import Detection, TrackedObject  # noqa: E402
import src.violations.triple_riding as triple_riding  # noqa: E402

WEIGHTS = ROOT / "models" / "weights"
VEHICLE_W = WEIGHTS / "yolo11s.pt"
HELMET_W = WEIGHTS / "helmet_yolov8.pt"
PLATE_W = WEIGHTS / "plate_yolov8.pt"
SEATBELT_W = WEIGHTS / "seatbelt.pt"

_VIOLATION_COLORS = {
    "helmet": (0, 0, 255),
    "triple_riding": (0, 165, 255),
}

st.set_page_config(page_title="Traffic Violation Detection", page_icon="🚦", layout="wide")


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@st.cache_resource(show_spinner="Loading detection models…")
def load_models(device: str):
    from src.detection.vehicle_detector import VehicleDetector
    from src.detection.plate_detector import PlateDetector
    from src.violations.helmet import HelmetChecker
    from src.ocr.plate_reader import PlateReader

    missing = [p.name for p in (VEHICLE_W, HELMET_W, PLATE_W) if not p.exists()]
    if missing:
        st.error(
            "Missing model weights: " + ", ".join(missing) +
            ". Commit them to models/weights/ (they are ~19 MB each)."
        )
        st.stop()

    vehicle = VehicleDetector(str(VEHICLE_W), conf_threshold=0.35, device=device)
    plate = PlateDetector(str(PLATE_W), conf_threshold=0.30, device=device)
    helmet = HelmetChecker(str(HELMET_W), conf_threshold=0.35, device=device)
    reader = PlateReader(use_gpu=(device != "cpu"))

    seatbelt = None
    if SEATBELT_W.exists():
        from ultralytics import YOLO
        seatbelt = YOLO(str(SEATBELT_W))
    return vehicle, plate, helmet, reader, seatbelt


def _tracks_from_detections(dets: list[Detection]) -> list[TrackedObject]:
    """Single-image 'tracks': one track per detection (no temporal tracking needed
    for helmet / triple-riding, which are per-frame rules)."""
    tracks = []
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d.bbox
        tracks.append(TrackedObject(
            track_id=i, class_name=d.class_name, bbox=d.bbox,
            confidence=d.confidence, frame_id=0,
            centroid_history=[((x1 + x2) // 2, (y1 + y2) // 2)],
        ))
    return tracks


def _annotate(frame, dets, violations, plates, seatbelt_boxes=()):
    img = frame.copy()
    # Vehicles / persons — light boxes.
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(img, d.class_name, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    # Violations — bold coloured boxes.
    for v in violations:
        x1, y1, x2, y2 = v.bbox
        color = _VIOLATION_COLORS.get(v.violation_type, (255, 0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"{v.violation_type.upper()} {v.confidence:.2f}"
        cv2.rectangle(img, (x1, y1 - 22), (x1 + 12 * len(label), y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # Plates — boxes + OCR text.
    for (bbox, text, conf) in plates:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        if text:
            cv2.putText(img, text, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    # Seatbelt-model raw detections — cyan (polarity TBD).
    for (bbox, conf) in seatbelt_boxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(img, f"SEATBELT? {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return img


def run_detection(frame, models):
    vehicle, plate, helmet, reader, seatbelt = models
    dets = vehicle.detect(frame, frame_id=0)
    tracks = _tracks_from_detections(dets)

    violations = []
    violations += triple_riding.check(tracks, frame_id=0)
    violations += helmet.check(tracks, frame, frame_id=0)

    plates = []
    for p in plate.detect(frame, frame_id=0):
        x1, y1, x2, y2 = p.bbox
        crop = frame[y1:y2, x1:x2]
        result = reader.read(crop)
        plates.append(((x1, y1, x2, y2), result.text if result else "", result.confidence if result else 0.0))

    # Seatbelt model raw detections (polarity TBD — shown to confirm what class 0 means).
    seatbelt_boxes = []
    if seatbelt is not None:
        sres = seatbelt.predict(source=frame, conf=0.30, device=vehicle.device, verbose=False)
        if sres and sres[0].boxes:
            for b in sres[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                seatbelt_boxes.append(((x1, y1, x2, y2), float(b.conf[0])))

    return dets, violations, plates, seatbelt_boxes


def main():
    st.title("🚦 Automated Traffic Violation Detection")
    st.caption("Upload a traffic image — the system detects road users, flags violations "
               "(helmet non-compliance, triple riding), and reads number plates.")

    device = _device()
    st.sidebar.markdown(f"**Compute:** `{device}`")
    st.sidebar.markdown(
        "**Models**\n- Vehicles/persons: YOLO11s (COCO)\n- Helmet: fine-tuned YOLO\n"
        "- Plate: fine-tuned YOLO + EasyOCR"
    )

    sample_dir = ROOT / "samples"
    sample_files = sorted([p for p in sample_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]) \
        if sample_dir.is_dir() else []

    col = st.columns([2, 1])
    uploaded = col[0].file_uploader("Upload a traffic image", type=["jpg", "jpeg", "png"])
    chosen_sample = None
    if sample_files:
        names = ["—"] + [p.name for p in sample_files]
        pick = col[1].selectbox("…or try a sample", names)
        if pick != "—":
            chosen_sample = sample_dir / pick

    frame = None
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif chosen_sample is not None:
        frame = cv2.imread(str(chosen_sample))

    if frame is None:
        st.info("Upload an image to begin.")
        return

    models = load_models(device)
    with st.spinner("Analysing…"):
        dets, violations, plates, seatbelt_boxes = run_detection(frame, models)
    annotated = _annotate(frame, dets, violations, plates, seatbelt_boxes)

    left, right = st.columns(2)
    left.subheader("Annotated evidence")
    left.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    right.subheader("Results")
    right.metric("Road users detected", len(dets))
    right.metric("Violations flagged", len(violations))
    if violations:
        right.dataframe(
            [{"type": v.violation_type, "confidence": round(v.confidence, 2), "status": v.status}
             for v in violations],
            use_container_width=True,
        )
    else:
        right.success("No violations detected in this image.")

    read_plates = [{"plate": t, "confidence": round(c, 2)} for (_b, t, c) in plates if t]
    if read_plates:
        right.subheader("Number plates")
        right.dataframe(read_plates, use_container_width=True)

    if seatbelt_boxes:
        right.subheader("Seatbelt model (cyan boxes)")
        right.caption("Calibration: confirm whether these boxes appear on belted or "
                      "un-belted drivers, then it becomes a seatbelt violation rule.")
        right.metric("Seatbelt-model detections", len(seatbelt_boxes))


if __name__ == "__main__":
    main()
