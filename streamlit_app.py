"""
Interactive demo for the Traffic Violation Detection System.

This is the public, CPU-friendly demo entry point (deploy target for
Streamlit Community Cloud / Hugging Face Spaces). Upload a traffic image or
short video clip and the app runs the real detectors + violation logic and
returns annotated evidence.

Visual violations detected (no camera calibration needed):
  - Helmet non-compliance (rider_no_helmet YOLO)
  - Triple riding (person-containment rule on motorcycle bbox)
  - License plate reading (YOLO + EasyOCR)

Geometry-based violations (stop-line, wrong-side, parking, red-light) require
per-camera calibration and run only in the full video pipeline (app.py --video).

Run locally:   streamlit run streamlit_app.py
Deploy:        see docs/DEPLOYMENT.md
"""

import sys
import tempfile
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

_VIOLATION_COLORS = {
    "helmet": (0, 0, 255),
    "triple_riding": (0, 165, 255),
}

# Maximum frames to sample from a video upload (keeps it fast on CPU cloud)
MAX_VIDEO_FRAMES = 30
# Process every Nth frame from the video
FRAME_SAMPLE_INTERVAL = 10

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
    return vehicle, plate, helmet, reader


def _tracks_from_detections(dets: list[Detection]) -> list[TrackedObject]:
    """Single-frame 'tracks': one track per detection (no temporal tracking
    needed for helmet / triple-riding which are per-frame rules)."""
    tracks = []
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d.bbox
        tracks.append(TrackedObject(
            track_id=i, class_name=d.class_name, bbox=d.bbox,
            confidence=d.confidence, frame_id=0,
            centroid_history=[((x1 + x2) // 2, (y1 + y2) // 2)],
        ))
    return tracks


def _annotate(frame, dets, violations, plates):
    img = frame.copy()
    # Vehicles / persons — light green boxes.
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
    return img


def run_detection(frame: np.ndarray, models, frame_id: int = 0):
    vehicle, plate, helmet, reader = models
    dets = vehicle.detect(frame, frame_id=frame_id)
    tracks = _tracks_from_detections(dets)

    violations = []
    violations += triple_riding.check(tracks, frame_id=frame_id)
    violations += helmet.check(tracks, frame, frame_id=frame_id)

    plates = []
    for p in plate.detect(frame, frame_id=frame_id):
        x1, y1, x2, y2 = p.bbox
        crop = frame[y1:y2, x1:x2]
        result = reader.read(crop)
        plates.append(((x1, y1, x2, y2), result.text if result else "", result.confidence if result else 0.0))

    return dets, violations, plates


def _show_results(dets, violations, plates, col):
    col.metric("Road users detected", len(dets))
    col.metric("Violations flagged", len(violations))
    if violations:
        col.dataframe(
            [{"type": v.violation_type, "confidence": round(v.confidence, 2), "status": v.status}
             for v in violations],
            use_container_width=True,
        )
    else:
        col.success("No violations detected.")

    read_plates = [{"plate": t, "confidence": round(c, 2)} for (_b, t, c) in plates if t]
    if read_plates:
        col.subheader("Number plates read")
        col.dataframe(read_plates, use_container_width=True)


def process_image(frame: np.ndarray, models):
    dets, violations, plates = run_detection(frame, models)
    annotated = _annotate(frame, dets, violations, plates)

    left, right = st.columns(2)
    left.subheader("Annotated evidence")
    left.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    right.subheader("Results")
    _show_results(dets, violations, plates, right)


def process_video(video_path: str, models):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Sample frames evenly across the video up to MAX_VIDEO_FRAMES
    sample_interval = max(FRAME_SAMPLE_INTERVAL, total_frames // MAX_VIDEO_FRAMES)

    st.info(
        f"Video: {total_frames} frames @ {video_fps:.0f} fps. "
        f"Sampling every {sample_interval} frames (≤{MAX_VIDEO_FRAMES} frames total)."
    )

    progress = st.progress(0, text="Analysing video…")
    frame_idx = 0
    processed = 0
    all_violations = []
    all_plates: dict[str, float] = {}
    annotated_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % sample_interval != 0:
            continue

        dets, violations, plates = run_detection(frame, models, frame_id=frame_idx)
        all_violations.extend(violations)
        for (_b, text, conf) in plates:
            if text and conf > all_plates.get(text, 0):
                all_plates[text] = conf

        annotated = _annotate(frame, dets, violations, plates)
        annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        processed += 1
        progress.progress(min(processed / MAX_VIDEO_FRAMES, 1.0), text=f"Frame {frame_idx}…")
        if processed >= MAX_VIDEO_FRAMES:
            break

    cap.release()
    progress.empty()

    st.success(f"Processed {processed} sampled frames.")

    # Show a carousel of sampled annotated frames
    st.subheader("Sampled annotated frames")
    cols = st.columns(min(3, len(annotated_frames)))
    for i, img in enumerate(annotated_frames):
        cols[i % 3].image(img, use_container_width=True, caption=f"Sample {i + 1}")

    # Summary
    st.subheader("Summary across all sampled frames")
    c1, c2 = st.columns(2)
    c1.metric("Total violations found", len(all_violations))
    c2.metric("Unique plates read", len(all_plates))

    if all_violations:
        st.dataframe(
            [{"type": v.violation_type, "confidence": round(v.confidence, 2), "status": v.status}
             for v in all_violations],
            use_container_width=True,
        )
    if all_plates:
        st.subheader("Plates detected")
        st.dataframe(
            [{"plate": k, "confidence": round(v, 2)} for k, v in all_plates.items()],
            use_container_width=True,
        )


def main():
    st.title("🚦 Automated Traffic Violation Detection")
    st.caption(
        "Upload a traffic **image or video** — the system detects road users, flags violations "
        "(helmet non-compliance, triple riding), and reads number plates."
    )

    device = _device()
    st.sidebar.markdown(f"**Compute:** `{device}`")
    st.sidebar.markdown(
        "**Models**\n- Vehicles/persons: YOLO11s (COCO)\n- Helmet: fine-tuned YOLO\n"
        "- Plate: fine-tuned YOLO + EasyOCR"
    )
    st.sidebar.markdown(
        "**Violations detected here**\n"
        "- 🪖 Helmet non-compliance\n"
        "- 🏍️ Triple riding\n"
        "- 🔢 License plate reading\n\n"
        "_Stop-line, wrong-side, parking and red-light require camera calibration "
        "and run only in the full local pipeline (`app.py --video`)._"
    )

    sample_dir = ROOT / "samples"
    sample_files = sorted([p for p in sample_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".mp4"}]) \
        if sample_dir.is_dir() else []

    tab_upload, tab_sample = st.tabs(["📤 Upload", "🖼️ Try a sample"])

    # ── Upload tab ──────────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a traffic image or video clip",
            type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        )
        if uploaded is not None:
            models = load_models(device)
            suffix = Path(uploaded.name).suffix.lower()

            if suffix in {".jpg", ".jpeg", ".png"}:
                data = np.frombuffer(uploaded.read(), np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                with st.spinner("Analysing image…"):
                    process_image(frame, models)

            elif suffix in {".mp4", ".avi", ".mov"}:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                with st.spinner("Analysing video…"):
                    process_video(tmp_path, models)

    # ── Sample tab ──────────────────────────────────────────────────────────
    with tab_sample:
        if not sample_files:
            st.info("No sample files found in the `samples/` folder.")
        else:
            names = [p.name for p in sample_files]
            pick = st.selectbox("Choose a sample", names)
            chosen = sample_dir / pick
            if st.button("▶️ Run detection on sample"):
                models = load_models(device)
                suffix = chosen.suffix.lower()
                if suffix in {".jpg", ".jpeg", ".png"}:
                    frame = cv2.imread(str(chosen))
                    with st.spinner("Analysing image…"):
                        process_image(frame, models)
                elif suffix in {".mp4", ".avi", ".mov"}:
                    with st.spinner("Analysing video…"):
                        process_video(str(chosen), models)


if __name__ == "__main__":
    main()
