"""
Main video processing pipeline.
Wires all modules together in order:
  Frame → Preprocess → Detect → Track → TrackMemory → Violations → ANPR → Evidence → DB

Optimizations (via TrackMemory):
  - Helmet YOLO only runs when at least one motorcycle track needs a (re-)check.
  - Seatbelt CNN only runs when a car track needs a check (indeterminate emitted once, not per frame).
  - Plate detection + OCR only runs on violation frames, and results are cached per track.
"""

import cv2
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_pipeline, load_violations, get_camera_config, load_tracker_config
from src.preprocessing.frame_processor import process_frame
from src.models import ViolationRecord

# Heavy ML imports are done lazily inside run() so that --dry-run works
# on machines without the full inference stack.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run(source: str, camera_id: str = "cam_001", dry_run: bool = False, show: bool = False):
    pipeline_cfg = load_pipeline()
    violation_cfg = load_violations()
    tracker_cfg = load_tracker_config()
    camera_cfg = get_camera_config(camera_id)

    if camera_cfg is None:
        log.error("Camera '%s' not found in cameras.yaml", camera_id)
        return

    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error("Cannot open video source: %s", source)
        return

    log.info("Pipeline starting | source=%s | camera=%s | dry_run=%s", source, camera_id, dry_run)

    target_fps = pipeline_cfg["input"]["target_fps"]
    resize_w = pipeline_cfg["input"].get("resize_width", 0)
    pre_cfg = pipeline_cfg["preprocessing"]
    inf_cfg = pipeline_cfg["inference"]
    ev_cfg = pipeline_cfg["evidence"]
    db_cfg = pipeline_cfg["database"]

    if not dry_run:
        from src.detection.vehicle_detector import VehicleDetector
        from src.detection.plate_detector import PlateDetector
        from src.tracking.tracker import Tracker
        from src.tracking.track_memory import TrackMemory
        from src.ocr.plate_reader import PlateReader
        from src.evidence.generator import EvidenceGenerator
        from src.database.schema import init_db
        from src.database.repository import insert_violation
        import src.violations.triple_riding as triple_riding
        import src.violations.wrong_side as wrong_side
        import src.violations.stop_line as stop_line_mod
        import src.violations.red_light as red_light_mod
        import src.violations.parking as parking_mod
        from src.violations.helmet import HelmetChecker
        from src.violations.seatbelt import SeatbeltChecker

        detector = VehicleDetector(
            inf_cfg["vehicle_detector"] if "vehicle_detector" in inf_cfg
            else pipeline_cfg["models"]["vehicle_detector"],
            conf_threshold=inf_cfg["vehicle_conf"],
            nms_iou=inf_cfg["nms_iou"],
            device=inf_cfg["device"],
        )
        plate_detector = PlateDetector(
            pipeline_cfg["models"]["plate_detector"],
            conf_threshold=inf_cfg["plate_conf"],
            device=inf_cfg["device"],
        )
        tracker = Tracker(
            track_thresh=tracker_cfg.get("track_thresh", 0.50),
            track_buffer=tracker_cfg.get("track_buffer", 30),
            match_thresh=tracker_cfg.get("match_thresh", 0.80),
        )
        memory = TrackMemory(
            helmet_refresh_interval=violation_cfg.get("helmet", {}).get("refresh_interval", 30),
            seatbelt_refresh_interval=violation_cfg.get("seatbelt", {}).get("refresh_interval", 60),
            min_helmet_confirm=violation_cfg.get("helmet", {}).get("min_confirm_frames", 2),
            min_seatbelt_confirm=violation_cfg.get("seatbelt", {}).get("min_confirm_frames", 2),
        )
        helmet_checker = HelmetChecker(
            pipeline_cfg["models"]["helmet_classifier"],
            conf_threshold=inf_cfg["helmet_conf"],
            device=inf_cfg["device"],
            head_roi_fraction=violation_cfg["helmet"]["head_roi_fraction"],
            flag_invalid_helmet=violation_cfg["helmet"].get("flag_invalid_helmet", False),
        )
        seatbelt_checker = SeatbeltChecker(
            model_path=pipeline_cfg["models"].get("seatbelt_classifier"),
            conf_threshold=inf_cfg["seatbelt_conf"],
            device=inf_cfg["device"],
            windshield_top_fraction=violation_cfg["seatbelt"]["windshield_top_fraction"],
            windshield_bottom_fraction=violation_cfg["seatbelt"]["windshield_bottom_fraction"],
            min_crop_width=violation_cfg["seatbelt"]["min_crop_width"],
            min_crop_height=violation_cfg["seatbelt"]["min_crop_height"],
        )
        plate_reader = PlateReader(use_gpu=(inf_cfg["device"] != "cpu"))
        evidence_gen = EvidenceGenerator(ev_cfg["save_dir"], ev_cfg["jpeg_quality"])
        engine = init_db(db_cfg["path"])

    # Camera geometry
    stop_line_pts = camera_cfg.get("stop_line", [[0, 0], [1, 0]])
    signal_roi = tuple(camera_cfg.get("signal_roi", [0, 0, 1, 1]))
    no_parking_zones = camera_cfg.get("no_parking_zones", [])
    allowed_dir = camera_cfg.get("allowed_direction_deg", 90)
    dir_tol = camera_cfg.get("direction_tolerance_deg", 30)

    park_cfg = violation_cfg.get("illegal_parking", {})
    ws_cfg = violation_cfg.get("wrong_side", {})
    sl_cfg = violation_cfg.get("stop_line", {})
    rl_cfg = violation_cfg.get("red_light", {})
    tr_cfg = violation_cfg.get("triple_riding", {})

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(video_fps / target_fps))
    frame_idx = 0
    processed_count = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue
        processed_count += 1

        if resize_w > 0 and frame.shape[1] != resize_w:
            scale = resize_w / frame.shape[1]
            frame = cv2.resize(frame, (resize_w, int(frame.shape[0] * scale)))

        processed_frame, quality = process_frame(
            frame,
            clahe_clip_limit=pre_cfg["clahe_clip_limit"],
            clahe_tile_size=pre_cfg["clahe_tile_size"],
            blur_threshold=pre_cfg["blur_threshold"],
            apply_rain_filter=pre_cfg["apply_rain_filter"],
        )

        if dry_run:
            if show:
                cv2.imshow("Dry Run", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        # Detection
        detections = detector.detect(processed_frame, frame_idx)

        # Tracking
        tracks = tracker.update(detections, processed_frame, frame_idx)

        # Sync track memory — evict entries for tracks the tracker has dropped.
        active_ids = {t.track_id for t in tracks}
        memory.evict_stale(active_ids)

        # Ensure every active track has a memory entry.
        for t in tracks:
            memory.get_or_create(t.track_id, t.class_name)

        # Violations
        all_violations: list[ViolationRecord] = []

        all_violations += triple_riding.check(
            tracks, frame_idx, camera_id,
            min_overlap_ratio=tr_cfg.get("min_person_overlap_ratio", 0.5),
        )
        all_violations += wrong_side.check(
            tracks, frame_idx,
            allowed_direction_deg=allowed_dir,
            direction_tolerance_deg=dir_tol,
            min_track_frames=ws_cfg.get("min_track_frames", 8),
            consecutive_wrong_frames=ws_cfg.get("consecutive_wrong_frames", 5),
            camera_id=camera_id,
        )
        all_violations += stop_line_mod.check(
            tracks, processed_frame, frame_idx,
            stop_line=stop_line_pts,
            signal_roi=signal_roi,
            crossing_margin_px=sl_cfg.get("crossing_margin_px", 5),
            red_pixel_fraction=rl_cfg.get("red_pixel_fraction", 0.15),
            green_pixel_fraction=rl_cfg.get("green_pixel_fraction", 0.10),
            camera_id=camera_id,
        )
        all_violations += red_light_mod.check(
            tracks, processed_frame, frame_idx,
            stop_line=stop_line_pts,
            signal_roi=signal_roi,
            red_pixel_fraction=rl_cfg.get("red_pixel_fraction", 0.15),
            green_pixel_fraction=rl_cfg.get("green_pixel_fraction", 0.10),
            camera_id=camera_id,
        )
        all_violations += parking_mod.check(
            tracks, frame_idx,
            no_parking_zones=no_parking_zones,
            dwell_time_seconds=park_cfg.get("dwell_time_seconds", 180),
            stationary_pixel_threshold=park_cfg.get("stationary_pixel_threshold", 15),
            camera_id=camera_id,
        )

        # ML-based violations use track memory to skip redundant inference.
        all_violations += helmet_checker.check(
            tracks, processed_frame, frame_idx, camera_id, track_memory=memory,
        )
        all_violations += seatbelt_checker.check(
            tracks, processed_frame, frame_idx, camera_id, track_memory=memory,
        )

        # ANPR + evidence for confirmed/review violations
        for v in all_violations:
            if v.status == "indeterminate":
                insert_violation(v, engine)
                continue

            v.is_blurry = quality.is_blurry

            # Plate caching: reuse OCR result if we already read this vehicle's plate.
            mem_state = memory.get(v.vehicle_id)
            if mem_state and mem_state.plate_number is not None:
                v.plate_number = mem_state.plate_number
                v.plate_confidence = mem_state.plate_confidence
            else:
                plate_dets = plate_detector.detect_in_vehicle_crop(processed_frame, v.bbox, frame_idx)
                if plate_dets:
                    best_plate = plate_dets[0]
                    px1, py1, px2, py2 = best_plate.bbox
                    plate_crop = processed_frame[py1:py2, px1:px2]
                    plate_result = plate_reader.read(plate_crop)
                    if plate_result:
                        v.plate_number = plate_result.text
                        v.plate_confidence = plate_result.confidence
                        if mem_state:
                            mem_state.plate_number = plate_result.text
                            mem_state.plate_confidence = plate_result.confidence

            v = evidence_gen.save(processed_frame, v)
            insert_violation(v, engine)

            log.info(
                "VIOLATION | type=%s status=%s conf=%.2f plate=%s frame=%d",
                v.violation_type, v.status, v.confidence, v.plate_number, frame_idx,
            )

        if show:
            display = _draw_tracks(processed_frame.copy(), tracks)
            cv2.imshow("Traffic Violation System", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    fps = processed_count / elapsed if elapsed > 0 else 0
    log.info("Done | frames_processed=%d | elapsed=%.1fs | throughput=%.1f fps", processed_count, elapsed, fps)


def _draw_tracks(frame, tracks) -> "np.ndarray":
    import numpy as np
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
        cv2.putText(frame, f"{t.class_name} #{t.track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    return frame
