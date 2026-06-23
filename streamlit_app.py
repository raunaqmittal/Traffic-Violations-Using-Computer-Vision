"""
Traffic Violation Detection System — Interactive Demo

Tabs:
  1. 🎯 Detect Violations  — upload image/video; runs helmet, seatbelt, triple riding, plates
                             + stop-line / parking if camera zones are configured
  2. 🗺️ Camera Setup       — configure stop line, signal ROI, and parking zones on a reference frame
  3. ℹ️  How It Works       — explainer for all 7 violation types
  4. 🖼️  Try a Sample       — pre-loaded test footage

Run locally:   streamlit run streamlit_app.py
Deploy:        see docs/DEPLOYMENT.md
"""

import math
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Streamlit Cloud mounts the app as a read-only filesystem except for /tmp.
# Ultralytics appends its own 'Ultralytics' subdir to YOLO_CONFIG_DIR,
# so setting it to /tmp results in the correct writable path: /tmp/Ultralytics.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models import Detection, TrackedObject  # noqa: E402
import src.components.violations.triple_riding as triple_riding  # noqa: E402

WEIGHTS = ROOT / "models" / "weights"
VEHICLE_W = WEIGHTS / "yolo11s.pt"
HELMET_W  = WEIGHTS / "helmet_yolov8.pt"
PLATE_W   = WEIGHTS / "plate_yolov8.pt"

_CLASS_COLORS = {
    "person":     (255,  50, 255),
    "motorcycle": (  0, 255, 255),
    "car":        (255, 255,   0),
    "truck":      (  0, 165, 255),
    "bus":        (  0, 100, 255),
    "bicycle":    (  0, 255,   0),
}
_VIOLATION_COLORS = {
    "helmet":       (  0,   0, 255),
    "seatbelt":     (255,   0,   0),
    "triple_riding":(  0, 165, 255),
    "stop_line":    (  0, 255, 255),
    "illegal_parking": (255, 0, 100),
}

MAX_VIDEO_FRAMES   = 30
FRAME_SAMPLE_STEP  = 10

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Violation Detection",
    page_icon="🚦",
    layout="wide",
)


# ─────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────
def _init_zones():
    if "zones" not in st.session_state:
        st.session_state.zones = {
            "stop_line":    None,   # [[x1,y1],[x2,y2]]
            "signal_roi":   None,   # [x,y,w,h]
            "parking_zones": [],    # [{"name":str, "polygon":[[x,y],...]}]
        }


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ─────────────────────────────────────────────
# Model loading (cached across reruns)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection models…")
def load_models(device: str):
    try:
        from src.components.detection.vehicle_detector import VehicleDetector
        from src.components.detection.plate_detector import PlateDetector
        from src.components.violations.helmet import HelmetChecker
        from src.components.violations.seatbelt import SeatbeltChecker
        from src.components.ocr.plate_reader import PlateReader
    except Exception as _e:
        import sys
        st.error(
            f"**Failed to load detection modules** — this usually means a Python version "
            f"mismatch on Streamlit Cloud.\n\n"
            f"**Fix:** Go to your app's ⚙️ Settings → Advanced → Python version → select **3.11**"
            f", then click *Save* and *Reboot*.\n\n"
            f"Error detail: `{_e}`"
        )
        st.stop()

    missing = [p.name for p in (VEHICLE_W, HELMET_W, PLATE_W) if not p.exists()]
    if missing:
        st.error("Missing model weights: " + ", ".join(missing))
        st.stop()

    vehicle  = VehicleDetector(str(VEHICLE_W), conf_threshold=0.35, device=device)
    plate    = PlateDetector(str(PLATE_W),  conf_threshold=0.01, device=device)
    helmet   = HelmetChecker(str(HELMET_W), conf_threshold=0.35, device=device)
    seatbelt = SeatbeltChecker(model_path=None, conf_threshold=0.55, device=device)
    reader   = PlateReader(use_gpu=(device != "cpu"))
    return vehicle, plate, helmet, reader, seatbelt


# ─────────────────────────────────────────────
# Detection helpers
# ─────────────────────────────────────────────
def _tracks_from_detections(dets: list[Detection]) -> list[TrackedObject]:
    tracks = []
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d.bbox
        tracks.append(TrackedObject(
            track_id=i, class_name=d.class_name, bbox=d.bbox,
            confidence=d.confidence, frame_id=0,
            centroid_history=[((x1 + x2) // 2, (y1 + y2) // 2)],
        ))
    return tracks


def _annotate(frame, dets, violations, plates, zones=None):
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        c = _CLASS_COLORS.get(d.class_name, (0, 200, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, d.class_name, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    for v in violations:
        x1, y1, x2, y2 = v.bbox
        c = _VIOLATION_COLORS.get(v.violation_type, (255, 0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
        label = f"{v.violation_type.upper()} {v.confidence:.2f}"
        cv2.rectangle(img, (x1, y1 - 24), (x1 + 14 * len(label), y1), c, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    for (bbox, text, conf) in plates:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
        if text:
            cv2.putText(img, text, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
    # Draw zone overlays
    if zones:
        sl = zones.get("stop_line")
        if sl:
            cv2.line(img, tuple(sl[0]), tuple(sl[1]), (0, 255, 0), 2)
            cv2.putText(img, "STOP LINE", (sl[0][0], sl[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for zone in zones.get("parking_zones", []):
            pts = np.array(zone["polygon"], dtype=np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 200), 2)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.putText(img, "NO PARK", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        direction = zones.get("allowed_direction")
        if direction is not None:
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            rad = math.radians(direction)
            length = min(w, h) // 4
            dx = int(math.cos(rad) * length)
            dy = int(math.sin(rad) * length)
            cv2.arrowedLine(img, (cx, cy), (cx + dx, cy + dy), (255, 0, 255), 4, tipLength=0.2)
            cv2.putText(img, f"LEGAL DIRECTION: {direction} deg", (cx - 50, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    return img


def run_visual_detection(frame: np.ndarray, models, frame_id: int = 0):
    """Helmet, seatbelt, triple riding, plates — no geometry needed."""
    vehicle, plate, helmet, reader, seatbelt = models
    dets   = vehicle.detect(frame, frame_id=frame_id)
    tracks = _tracks_from_detections(dets)

    raw_violations = []
    raw_violations += triple_riding.check(tracks, frame_id=frame_id)
    raw_violations += helmet.check(tracks, frame, frame_id=frame_id)
    raw_violations += seatbelt.check(tracks, frame, frame_id=frame_id)

    # Filter: skip indeterminate seatbelt; skip seatbelt on bike riders
    motorcycles = [t for t in tracks if t.class_name in ("motorcycle", "bicycle")]
    violations = []
    for v in raw_violations:
        if v.violation_type == "seatbelt":
            if v.status == "indeterminate":
                continue
            cx = (v.bbox[0] + v.bbox[2]) / 2
            cy = (v.bbox[1] + v.bbox[3]) / 2
            if any(m.bbox[0] <= cx <= m.bbox[2] and m.bbox[1] <= cy <= m.bbox[3] for m in motorcycles):
                continue
        violations.append(v)

    plates = []
    for p in plate.detect(frame, frame_id=frame_id):
        x1, y1, x2, y2 = p.bbox
        crop   = frame[y1:y2, x1:x2]
        result = reader.read(crop)
        plates.append(((x1, y1, x2, y2), result.text if result else "", result.confidence if result else 0.0))

    return dets, violations, plates


# ─────────────────────────────────────────────
# Geometry violation helpers
# ─────────────────────────────────────────────
def _line_side(pt, p1, p2) -> float:
    """Signed cross product — which side of p1→p2 is pt on."""
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


def _point_in_polygon(pt, polygon) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def check_stop_line_violations(sampled_detections, stop_line):
    """
    Returns list of (frame_idx, vehicle_class, bbox) that crossed the stop line.
    Uses bottom-center of vehicle bbox as the crossing point.
    Only flags vehicles (not persons) past the line.
    """
    results = []
    vehicle_classes = {"car", "truck", "bus", "motorcycle", "auto-rickshaw"}
    p1, p2 = stop_line[0], stop_line[1]
    for frame_idx, dets in sampled_detections:
        for d in dets:
            if d.class_name not in vehicle_classes:
                continue
            cx = (d.bbox[0] + d.bbox[2]) // 2
            cy = d.bbox[3]  # bottom of vehicle
            if _line_side((cx, cy), p1, p2) > 0:
                results.append((frame_idx, d.class_name, d.bbox))
    return results


def check_parking_violations(sampled_detections, parking_zones, min_frames: int = 2):
    """
    Returns vehicles that appear inside a no-parking zone in >= min_frames sampled frames.
    Uses centroid proximity (50px) to link the same vehicle across sampled frames.
    """
    vehicle_classes = {"car", "truck", "bus", "motorcycle"}
    # track_pool: list of {"centroid": (cx, cy), "bbox": bbox, "zone": name, "count": int}
    tracks = []
    results = []

    for _frame_idx, dets in sampled_detections:
        for d in dets:
            if d.class_name not in vehicle_classes:
                continue
            cx = (d.bbox[0] + d.bbox[2]) // 2
            cy = (d.bbox[1] + d.bbox[3]) // 2
            for zone in parking_zones:
                if not _point_in_polygon((cx, cy), zone["polygon"]):
                    continue
                # Match to an existing track by centroid proximity
                matched = None
                for t in tracks:
                    if t["zone"] == zone["name"]:
                        dx = cx - t["centroid"][0]
                        dy = cy - t["centroid"][1]
                        if math.sqrt(dx * dx + dy * dy) < 80:
                            matched = t
                            break
                if matched:
                    matched["count"] += 1
                    matched["centroid"] = (cx, cy)
                    matched["bbox"] = d.bbox
                    if matched["count"] == min_frames:
                        results.append((zone["name"], d.class_name, d.bbox))
                else:
                    tracks.append({
                        "centroid": (cx, cy), "bbox": d.bbox,
                        "zone": zone["name"], "count": 1,
                    })
    return results


def check_wrong_side_violations(sampled_detections, allowed_direction_deg, min_frames: int = 2):
    """
    Returns vehicles moving against the allowed direction vector.
    Requires linking centroids across frames to determine motion vector.
    """
    vehicle_classes = {"car", "truck", "bus", "motorcycle", "auto-rickshaw"}
    tracks = []
    results = []
    
    rad = math.radians(allowed_direction_deg)
    allowed_vec = (math.cos(rad), math.sin(rad))

    for frame_idx, dets in sampled_detections:
        for d in dets:
            if d.class_name not in vehicle_classes:
                continue
            cx = (d.bbox[0] + d.bbox[2]) / 2
            cy = (d.bbox[1] + d.bbox[3]) / 2
            
            matched = None
            for t in tracks:
                dx = cx - t["centroids"][-1][0]
                dy = cy - t["centroids"][-1][1]
                if math.sqrt(dx * dx + dy * dy) < 100:  # generous tracking radius
                    matched = t
                    break
                    
            if matched:
                matched["centroids"].append((cx, cy))
                matched["bbox"] = d.bbox
                if len(matched["centroids"]) >= min_frames and not matched.get("flagged"):
                    first_pt = matched["centroids"][0]
                    last_pt = matched["centroids"][-1]
                    vec_x = last_pt[0] - first_pt[0]
                    vec_y = last_pt[1] - first_pt[1]
                    mag = math.sqrt(vec_x**2 + vec_y**2)
                    if mag > 20: # needs to have actually moved
                        vec_x /= mag
                        vec_y /= mag
                        dot = vec_x * allowed_vec[0] + vec_y * allowed_vec[1]
                        # If dot product is negative, it's moving in the opposite direction
                        if dot < -0.5: 
                            results.append((frame_idx, d.class_name, d.bbox))
                            matched["flagged"] = True
            else:
                tracks.append({"centroids": [(cx, cy)], "bbox": d.bbox, "class": d.class_name})
    return results


def _show_results(dets, violations, plates, col):
    col.metric("Road users detected", len(dets))
    col.metric("Violations flagged", len(violations))
    if violations:
        col.dataframe(
            [{"type": v.violation_type, "conf": round(v.confidence, 2), "status": v.status}
             for v in violations],
            use_container_width=True,
        )
    else:
        col.success("No violations detected.")
    read_plates = [{"plate": t, "confidence": round(c, 2)} for (_b, t, c) in plates if t]
    if read_plates:
        col.subheader("Plates read")
        col.dataframe(read_plates, use_container_width=True)


# ─────────────────────────────────────────────
# Tab 1: Detect Violations
# ─────────────────────────────────────────────
def tab_detect(models, device):
    st.header("🎯 Detect Violations")
    zones = st.session_state.get("zones", {})
    has_zones = bool(zones.get("stop_line") or zones.get("parking_zones"))

    if has_zones:
        st.info("✅ Camera zones configured — stop-line and parking violations will also be checked.")
    else:
        st.info("ℹ️ No camera zones configured. Set them in the **🗺️ Camera Setup** tab to enable "
                "stop-line and parking checks.")

    uploaded = st.file_uploader(
        "Upload a traffic image or video clip",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    )
    if uploaded is None:
        return

    suffix = Path(uploaded.name).suffix.lower()

    if suffix in {".jpg", ".jpeg", ".png"}:
        data  = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        with st.spinner("Analysing image…"):
            dets, violations, plates = run_visual_detection(frame, models)
        annotated = _annotate(frame, dets, violations, plates, zones=zones)
        left, right = st.columns(2)
        left.subheader("Annotated evidence")
        left.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        right.subheader("Results")
        _show_results(dets, violations, plates, right)

    elif suffix in {".mp4", ".avi", ".mov"}:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        _process_video(tmp_path, models, zones)


def _process_video(video_path: str, models, zones):
    cap = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    step   = max(FRAME_SAMPLE_STEP, total // MAX_VIDEO_FRAMES)

    st.info(f"Video: {total} frames @ {fps:.0f} fps. Sampling every {step} frames.")

    progress    = st.progress(0, text="Analysing…")
    all_violations = []
    all_plates: dict[str, float] = {}
    annotated_frames = []
    sampled_detections = []  # for geometry checks: [(frame_idx, dets), ...]

    frame_idx = processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue

        dets, violations, plates = run_visual_detection(frame, models, frame_id=frame_idx)
        sampled_detections.append((frame_idx, dets))
        all_violations.extend(violations)
        for (_, text, conf) in plates:
            if text and conf > all_plates.get(text, 0):
                all_plates[text] = conf

        annotated = _annotate(frame, dets, violations, plates, zones=zones)
        annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        processed += 1
        progress.progress(min(processed / MAX_VIDEO_FRAMES, 1.0), text=f"Frame {frame_idx}…")
        if processed >= MAX_VIDEO_FRAMES:
            break

    cap.release()
    progress.empty()

    # ── Geometry violations ──────────────────────────────────────────────
    stop_violations  = []
    park_violations  = []
    wrong_violations = []
    if zones.get("stop_line"):
        stop_violations = check_stop_line_violations(sampled_detections, zones["stop_line"])
    if zones.get("parking_zones"):
        park_violations = check_parking_violations(sampled_detections, zones["parking_zones"])
    if zones.get("allowed_direction") is not None:
        wrong_violations = check_wrong_side_violations(sampled_detections, zones["allowed_direction"])

    st.success(f"Processed {processed} sampled frames.")

    # Annotated frame carousel
    st.subheader("Sampled annotated frames")
    n_cols = min(3, len(annotated_frames))
    if n_cols:
        cols = st.columns(n_cols)
        for i, img in enumerate(annotated_frames):
            cols[i % n_cols].image(img, use_container_width=True, caption=f"Frame sample {i + 1}")

    # Summary
    st.subheader("Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Visual violations", len(all_violations))
    c2.metric("Unique plates", len(all_plates))
    c3.metric("Stop-line violations", len(stop_violations))
    c4.metric("Parking violations", len(park_violations))
    c5.metric("Wrong-side violations", len(wrong_violations))

    if all_violations:
        st.markdown("**Visual violations (helmet / seatbelt / triple riding)**")
        st.dataframe(
            [{"type": v.violation_type, "conf": round(v.confidence, 2), "status": v.status}
             for v in all_violations],
            use_container_width=True,
        )
    if stop_violations:
        st.markdown("**Stop-line crossings detected**")
        st.dataframe(
            [{"frame": f, "vehicle": cl} for (f, cl, _) in stop_violations],
            use_container_width=True,
        )
    if park_violations:
        st.markdown("**Parking violations detected**")
        st.dataframe(
            [{"zone": z, "vehicle": cl} for (z, cl, _) in park_violations],
            use_container_width=True,
        )
    if wrong_violations:
        st.markdown("**Wrong-side driving detected**")
        st.dataframe(
            [{"frame": f, "vehicle": cl} for (f, cl, _) in wrong_violations],
            use_container_width=True,
        )
    if all_plates:
        st.markdown("**Plates detected**")
        st.dataframe(
            [{"plate": k, "confidence": round(v, 2)} for k, v in all_plates.items()],
            use_container_width=True,
        )


# ─────────────────────────────────────────────
# Tab 2: Camera Setup
# ─────────────────────────────────────────────
def _draw_zone_preview(frame, zones):
    img = frame.copy()
    sl = zones.get("stop_line")
    if sl:
        cv2.line(img, tuple(sl[0]), tuple(sl[1]), (0, 255, 0), 3)
        cv2.putText(img, "STOP LINE", (sl[0][0], sl[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    sr = zones.get("signal_roi")
    if sr:
        x, y, w, h = sr
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.putText(img, "SIGNAL ROI", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    for zone in zones.get("parking_zones", []):
        pts = np.array(zone["polygon"], dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 220), 3)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(img, f"NO PARK: {zone['name']}", (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
    direction = zones.get("allowed_direction")
    if direction is not None:
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        rad = np.radians(direction)
        length = min(w, h) // 4
        dx = int(np.cos(rad) * length)
        dy = int(np.sin(rad) * length)
        cv2.arrowedLine(img, (cx, cy), (cx + dx, cy + dy), (255, 0, 255), 4, tipLength=0.2)
        cv2.putText(img, f"LEGAL DIRECTION: {direction} deg", (cx - 50, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    return img


def _parse_polygon(text: str):
    """Parse 'x1,y1 x2,y2 x3,y3 ...' into [[x1,y1],[x2,y2],...]."""
    try:
        pts = []
        for pair in text.strip().split():
            x, y = pair.split(",")
            pts.append([int(x), int(y)])
        return pts if len(pts) >= 3 else None
    except Exception:
        return None


def tab_camera_setup():
    st.header("🗺️ Camera Zone Setup")
    st.markdown(
        "Define zones on a reference frame from your camera. "
        "These zones enable **stop-line**, **signal/red-light**, **parking**, and **wrong-side** violation detection."
    )

    ref_file = st.file_uploader(
        "Upload a clear reference frame from your camera (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="zone_ref",
    )

    zones = st.session_state.zones

    if ref_file:
        data      = np.frombuffer(ref_file.read(), np.uint8)
        ref_frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        h, w      = ref_frame.shape[:2]

        st.markdown(f"Frame size: **{w} × {h} px**. Use pixel coordinates below.")

        left, right = st.columns([2, 1])

        with right:
            # ── Stop Line ───────────────────────────────────────────────
            st.subheader("1️⃣ Stop Line")
            st.caption("Draw a line across the lane vehicles must not cross when red.")
            c1, c2 = st.columns(2)
            sl_x1 = c1.number_input("X1", 0, w, w // 4,   key="sl_x1")
            sl_y1 = c2.number_input("Y1", 0, h, h // 2,   key="sl_y1")
            sl_x2 = c1.number_input("X2", 0, w, 3 * w // 4, key="sl_x2")
            sl_y2 = c2.number_input("Y2", 0, h, h // 2,   key="sl_y2")
            if st.button("✅ Set stop line"):
                zones["stop_line"] = [[sl_x1, sl_y1], [sl_x2, sl_y2]]
                st.success("Stop line saved.")

            st.divider()

            # ── Signal ROI ──────────────────────────────────────────────
            st.subheader("2️⃣ Signal ROI")
            st.caption("Rectangle around the traffic signal light (for red/green detection).")
            c1, c2 = st.columns(2)
            sr_x = c1.number_input("X",      0, w, 10,  key="sr_x")
            sr_y = c2.number_input("Y",      0, h, 10,  key="sr_y")
            sr_w = c1.number_input("Width",  1, w, 50,  key="sr_w")
            sr_h = c2.number_input("Height", 1, h, 100, key="sr_h")
            if st.button("✅ Set signal ROI"):
                zones["signal_roi"] = [sr_x, sr_y, sr_w, sr_h]
                st.success("Signal ROI saved.")

            st.divider()

            # ── Parking Zones ────────────────────────────────────────────
            st.subheader("3️⃣ No-Parking Zones")
            st.caption("Enter polygon points as `x,y` pairs separated by spaces. Minimum 3 points.")
            zone_name = st.text_input("Zone name", value="Zone_1", key="pz_name")
            zone_pts  = st.text_area(
                "Polygon points (x,y pairs)",
                placeholder="100,400 300,400 300,600 100,600",
                key="pz_pts",
            )
            col_add, col_clear = st.columns(2)
            if col_add.button("➕ Add zone"):
                pts = _parse_polygon(zone_pts)
                if pts:
                    zones["parking_zones"].append({"name": zone_name, "polygon": pts})
                    st.success(f"Zone '{zone_name}' added ({len(pts)} points).")
                else:
                    st.error("Need at least 3 valid x,y points.")
            if col_clear.button("🗑️ Clear all zones"):
                zones["stop_line"]     = None
                zones["signal_roi"]    = None
                zones["parking_zones"] = []
                zones["allowed_direction"] = None
                st.info("All zones cleared.")

            st.divider()
            
            # ── Legal Traffic Direction ──────────────────────────────────────────
            st.subheader("4️⃣ Legal Traffic Direction")
            st.caption("Angle (in degrees) that vehicles are allowed to travel. (0=Right, 90=Down, 180=Left, 270=Up)")
            c1, c2 = st.columns([3, 1])
            direction_deg = c1.slider("Allowed Direction", 0, 359, 180, key="dir_deg")
            if c2.button("✅ Set Direction"):
                zones["allowed_direction"] = direction_deg
                st.success(f"Direction {direction_deg}° saved.")

            # Zone status summary
            st.divider()
            st.subheader("Current zones")
            st.write("Stop line:", "✅" if zones["stop_line"] else "❌ not set")
            st.write("Signal ROI:", "✅" if zones["signal_roi"] else "❌ not set")
            st.write("Legal Direction:", f"✅ ({zones['allowed_direction']}°)" if zones.get("allowed_direction") is not None else "❌ not set")
            for z in zones["parking_zones"]:
                st.write(f"Parking zone — {z['name']}: ✅ ({len(z['polygon'])} pts)")

        with left:
            preview = _draw_zone_preview(ref_frame, zones)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                     caption="Zone preview (updates after each save)", use_container_width=True)

    else:
        # Show current zone status even without uploading a frame
        st.info("Upload a reference frame to define and preview zones.")
        if any([zones.get("stop_line"), zones.get("signal_roi"), zones.get("parking_zones"), zones.get("allowed_direction") is not None]):
            st.markdown("**Zones already configured in this session:**")
            st.json(zones)

    st.markdown(
        "> **Tip:** Run `python scripts/draw_zones.py --video data/samples/your_video.mp4` "
        "locally to click-to-define zones and get the exact pixel coordinates."
    )


# ─────────────────────────────────────────────
# Tab 3: How It Works
# ─────────────────────────────────────────────
def tab_how_it_works():
    st.header("ℹ️ How the System Works")
    st.markdown(
        "This system processes every video frame through an 8-stage pipeline. "
        "The cloud demo runs **Stages 1–5** (visual violations). "
        "All 7 violations run in the full local pipeline (`app.py --video`)."
    )

    stages = [
        ("1. Image Preprocessing", "frame_processor.py",
         "Every frame is enhanced with **CLAHE** (contrast equalisation on the luminance channel) "
         "to improve detection in shadows and at night. A **Laplacian variance score** detects "
         "motion blur — blurry frames are still processed but evidence is tagged `is_blurry=True`."),
        ("2. Vehicle & Road User Detection", "vehicle_detector.py — YOLO11s",
         "A YOLO11s model pretrained on COCO scans the full frame. "
         "Detected classes: `car`, `truck`, `bus`, `motorcycle`, `person`. "
         "Each detection has a bounding box and a confidence score."),
        ("3. IoU Tracker", "tracker.py",
         "Each vehicle is assigned a stable ID that persists across frames, even through brief "
         "occlusions (track buffer = 30 frames). The tracker records the centroid history "
         "(where the vehicle has been) to enable direction-based violations."),
        ("4. TrackMemory", "track_memory.py",
         "Stores helmet, seatbelt, and plate results per track ID. Avoids running the heavy "
         "ML models every frame — helmet is re-checked every 3 sec, seatbelt every 6 sec. "
         "**Multi-frame confirmation:** a violation is only emitted after 2 separate recheck "
         "cycles agree (prevents false positives from a single blurry frame)."),
        ("5. Violation Engine — 7 violations", "src/violations/",
         "See the detailed breakdown below."),
        ("6. Confidence Routing", "classifier.py",
         "Every violation gets a status: `auto_flagged` (conf ≥ threshold) or `review` "
         "(conf < threshold). Indeterminate results (e.g. seatbelt on small crop) go straight "
         "to the review queue."),
        ("7. ANPR", "plate_detector.py + plate_reader.py (EasyOCR)",
         "Only triggered when a vehicle commits a confirmed or review violation — not every frame. "
         "Plate text is cached per track ID so OCR doesn't repeat for multi-violation vehicles."),
        ("8. Evidence + Database", "evidence/generator.py + database/",
         "An annotated JPEG and a JSON sidecar are saved for every violation. "
         "All records go into SQLite (`violations.db`) and are searchable via the local "
         "Streamlit analytics dashboard (`streamlit run dashboard/app.py`)."),
    ]

    for title, module, desc in stages:
        with st.expander(f"**{title}** — `{module}`"):
            st.markdown(desc)

    st.divider()
    st.subheader("The 7 Violations")

    violations_info = [
        ("🪖 Helmet Non-Compliance", "ML Model", "helmet_yolov8.pt",
         "A fine-tuned YOLO detector scans the full frame for bare heads. "
         "Head detections are spatially associated to motorcycle bounding boxes "
         "(search area extends 1.2× above the bike box to catch the rider's head). "
         "**Requires 2 confirmation cycles before logging.**"),
        ("🚗 Seatbelt Non-Compliance", "ML Model", "seatbelt_yolov11s.pt (HuggingFace)",
         "Only runs on `car`, `truck`, `bus` tracks. Crops the windshield ROI "
         "(top 15%–55% of the vehicle bbox) and classifies it as `seatbelt` or `no_seatbelt`. "
         "If the crop is too small the result is `indeterminate` (human review queue). "
         "**Requires 2 confirmation cycles. Pedestrians are explicitly excluded.**"),
        ("🏍️ Triple Riding", "Rule-based", "—",
         "For each motorcycle bbox, counts how many `person` bboxes are ≥50% contained inside it. "
         "If 3 or more → violation (confidence = 1.0). Uses *containment* not IoU, "
         "because riders sit above the COCO motorcycle box."),
        ("⬅️ Wrong-Side Driving", "Rule-based + Tracker", "cameras.yaml",
         "Computes the angle of the vehicle's centroid displacement vector over recent frames. "
         "If the angle deviates from the camera's `allowed_direction_deg` by more than the "
         "tolerance for 5 consecutive frames → violation. Ignores stationary vehicles."),
        ("🛑 Stop-Line Violation", "Rule-based + Signal", "cameras.yaml",
         "Checks if a vehicle's bottom-center crosses the virtual stop line "
         "while the traffic signal ROI shows a red pixel fraction ≥ 15%."),
        ("🔴 Red-Light Violation", "Rule-based + Signal", "cameras.yaml",
         "Variant of stop-line: flags vehicles already past the stop line when the signal turns red "
         "(i.e., they crossed just as the light changed)."),
        ("🅿️ Illegal Parking", "Rule-based + Tracker", "cameras.yaml",
         "No-parking zones are polygons defined in cameras.yaml. "
         "If a vehicle's centroid stays inside a polygon and moves < 15px for 3 minutes "
         "(dwell timer tracked per track ID) → violation."),
    ]

    for title, vtype, model, desc in violations_info:
        with st.expander(f"**{title}** — {vtype}"):
            st.markdown(f"**Model/Source:** `{model}`")
            st.markdown(desc)


# ─────────────────────────────────────────────
# Tab 4: Try a Sample
# ─────────────────────────────────────────────
def tab_sample(models):
    st.header("🖼️ Try a Sample")
    st.markdown(
        "This pre-loaded scene was chosen to demonstrate **every violation type** the system can detect. "
        "Zone boundaries (stop line and no-parking area) are pre-configured for you — just click **Run detection**."
    )

    sample_dir  = ROOT / "data" / "samples"
    scene_path  = sample_dir / "sample_scene.jpg"

    if not scene_path.exists():
        # Fallback: pick any image in the folder
        sample_files = sorted([
            p for p in sample_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]) if sample_dir.is_dir() else []
        if not sample_files:
            st.info("No sample files found in `data/samples/`.")
            return
        scene_path = sample_files[0]

    # Show legend
    with st.expander("📋 What to look for in this scene", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "- 🪖 **Helmet violation** — rider with no helmet (cyan box)\n"
                "- 🚗 **Seatbelt violation** — driver without seatbelt strap (blue box)\n"
                "- 🏍️ **Triple riding** — 3+ people on a motorcycle (orange box)\n"
            )
        with col2:
            st.markdown(
                "- 🛑 **Stop-line** — vehicle past the green line while signal is red\n"
                "- 🅿️ **Illegal parking** — vehicle inside the red NO PARK polygon\n"
                "- 🔢 **License plate** — plate text read by EasyOCR (yellow box)\n"
            )

    st.image(str(scene_path), caption="Sample traffic scene (pre-configured zones active)", use_container_width=True)

    # Pre-configured zones matched to the sample image geometry
    SAMPLE_ZONES = {
        "stop_line":    [[50, 420], [750, 420]],
        "signal_roi":   [680, 60, 80, 100],
        "parking_zones": [
            {"name": "No Parking Zone", "polygon": [[580, 340], [760, 340], [760, 500], [580, 500]]}
        ],
    }

    if st.button("▶️ Run detection on sample", type="primary", use_container_width=True):
        frame = cv2.imread(str(scene_path))
        if frame is None:
            st.error("Could not read the sample image.")
            return
        with st.spinner("Running models…"):
            dets, violations, plates = run_visual_detection(frame, models)
        annotated = _annotate(frame, dets, violations, plates, zones=SAMPLE_ZONES)
        left, right = st.columns([3, 2])
        left.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        right.subheader("Results")
        _show_results(dets, violations, plates, right)
        if not violations:
            right.info("No violations detected — try uploading your own image in the **Detect** tab!")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    _init_zones()
    device = _device()

    st.title("🚦 Automated Traffic Violation Detection")
    st.caption(
        "Upload traffic footage to detect violations. Configure camera zones to enable "
        "stop-line and parking checks in addition to helmet, seatbelt, and triple riding."
    )

    # Sidebar
    st.sidebar.markdown(f"**Compute:** `{device}`")
    st.sidebar.markdown(
        "**Visual violations** (cloud & local)\n"
        "- 🪖 Helmet non-compliance\n"
        "- 🚗 Seatbelt non-compliance\n"
        "- 🏍️ Triple riding\n"
        "- 🔢 License plate reading\n\n"
        "**With camera zones** (configure in Camera Setup tab)\n"
        "- 🛑 Stop-line violation\n"
        "- 🅿️ Illegal parking\n\n"
        "**Full local pipeline only**\n"
        "- ⬅️ Wrong-side driving\n"
        "- 🔴 Red-light (needs signal ROI)\n\n"
        "_Run `app.py --video` for the complete pipeline with IoU tracking._"
    )

    zones_configured = bool(
        st.session_state.zones.get("stop_line") or
        st.session_state.zones.get("parking_zones")
    )
    if zones_configured:
        st.sidebar.success("✅ Camera zones active")
    else:
        st.sidebar.warning("⚠️ No zones — set them in Camera Setup")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Detect Violations",
        "🗺️ Camera Setup",
        "ℹ️ How It Works",
        "🖼️ Try a Sample",
    ])

    models = load_models(device)

    with tab1:
        tab_detect(models, device)
    with tab2:
        tab_camera_setup()
    with tab3:
        tab_how_it_works()
    with tab4:
        tab_sample(models)


if __name__ == "__main__":
    main()
