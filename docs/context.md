# Project Context — Traffic Violation Detection System

> **Purpose**: Paste into any new AI chat to restore full project context instantly. Kept current as the project evolves.

---

## What This Project Is

A real-world deployable prototype for **Automated Photo Identification and Classification of Traffic Violations using Computer Vision** (AI/ML competition project — must be demonstrable and deployable).

- End-to-end pipeline: video/camera input → preprocessing → detection → tracking → violation checks → OCR → evidence → database → dashboard
- Stack: Python 3.11, YOLO11s (Ultralytics), self-contained IoU tracker, EasyOCR, OpenCV, SQLite/SQLAlchemy, Streamlit, Docker

---

## Current Status (2026-06-23)

**Working end-to-end on GPU.** Full pipeline verified on a real traffic video on RTX 3050 Ti (Lenovo Legion Slim 7i, i7 12th Gen, 16GB DDR5). Violations firing: `illegal_parking` and `stop_line` confirmed. Pipeline is now optimized with `TrackMemory` for heavy ML caching. 36/36 unit tests pass.

| Component | State |
|-----------|-------|
| Environment | Python **3.11.9** venv at `venv/`. System Python is 3.13 — **do NOT use it** (torch has no 3.13 wheels). torch 2.5.1+cu121, CUDA working on RTX 3050 Ti. |
| Vehicle/person model | `models/weights/yolo11s.pt` — COCO pretrained ✅ |
| Helmet model | `models/weights/helmet_yolov8.pt` — **TRAINED on Colab T4** ✅ |
| Plate model | `models/weights/plate_yolov8.pt` — **TRAINED on Colab T4** ✅ |
| Seatbelt model | `models/weights/seatbelt_yolov11s.pt` — **RISEF/yolov11s-seatbelt from HuggingFace** ✅ auto-downloaded |
| OCR | EasyOCR (GPU via torch) ✅ |
| Docker | `Dockerfile` + `docker-compose.yml` present — CPU image, GPU note inside |
| Cloud Demo | `streamlit_app.py` — 4-tab interactive UI (Detect, Camera Setup, How It Works, Try Sample) |
| Tests | 36/36 passing (`pytest tests/ -q`) |
| Git | `main` branch on `github.com/raunaqmittal/Traffic-Violations-Using-Computer-Vision` |

**Helmet model metrics (best.pt — epoch 54):**
- Overall: mAP@50 = 0.578, Precision = 0.61, Recall = 0.51
- `rider_no_helmet` class: **mAP@50 = 0.88, P = 0.83, R = 0.80** (strong — the violation class)
- Low overall dragged by a broken 1-sample `rider` class — not worth retraining

**Open items:**
- Capture plate model mAP (not done yet)
- Get a real Indian traffic video with riders and plates for a fuller demo (or use `scripts/images_to_video.py` on helmet dataset images)
- Optionally build `notebooks/03_evaluation.ipynb`

---

## Violations in Scope (All 7 — Nothing Dropped)

| Violation | Detection Method |
|-----------|-----------------|
| Helmet non-compliance | Full-frame helmet YOLO → `rider_no_helmet` heads associated to motorcycle tracks via upward-expanded bbox containment. NOT a head-crop (COCO motorcycle boxes exclude the rider's head). |
| Seatbelt non-compliance | YOLO11s classifier (`seatbelt_yolov11s.pt`) on windshield crop of **car/truck/bus bbox only** — pedestrians explicitly excluded. Marks `indeterminate` if crop too small. Requires 2 confirm cycles before violation is emitted. |
| Triple riding | Rule: count persons whose bbox is ≥ 50% **contained** (not IoU) in motorcycle bbox; ≥ 3 → violation |
| Wrong-side driving | Rule: centroid direction vector vs `allowed_direction_deg` for N consecutive frames |
| Stop-line violation | Rule: vehicle centroid past virtual line when signal is red/unknown |
| Red-light violation | Rule: confirmed red signal (HSV) + moving vehicle past stop line |
| Illegal parking | Rule: vehicle centroid inside no-parking polygon + stationary ≥ 3 min (configurable) |

**Excluded (never in problem statement):** mobile phone detection, lane violation.

---

## Full System Architecture

```
Video/Image Input
      ↓
Image Preprocessing  (CLAHE low-light, blur detection via Laplacian variance, rain filter)
      ↓
Vehicle & Road User Detection  (YOLO11s, COCO — car, truck, bus, motorcycle, person)
      ↓
IoU Tracker  (self-contained greedy IoU — stable IDs + 60-frame centroid history)
      ↓
TrackMemory Manager (caches ML inferences per track ID; rechecks on schedule or low confidence;
                     multi-frame confirmation before violation is emitted)
      ↓
Violation Detection Engine
  ├── Helmet         → full-frame helmet YOLO (cached per bike, min 2 confirm cycles)
  ├── Seatbelt       → YOLO11s seatbelt classifier (car/truck/bus only, min 2 confirm cycles)
  ├── Triple Riding  → person-containment rule (intersection / person-area)
  ├── Wrong-side     → direction vector rule on centroid history
  ├── Stop-line      → virtual line + signal state (HSV)
  ├── Red-light      → signal ROI (HSV) + vehicle position
  └── Illegal Parking→ polygon containment + dwell timer
      ↓
Violation Classifier & Confidence Scorer
  (≥ threshold → auto_flagged | below → review | unusable → indeterminate)
      ↓
License Plate Detection + EasyOCR (triggered only on violation, result cached per vehicle)
      ↓
Evidence Generator  (annotated JPEG + JSON sidecar per violation)
      ↓
SQLite Database  (SQLAlchemy — swap connection string for Postgres)
      ↓
Streamlit Dashboard  (KPIs, charts, searchable table, image viewer, CSV export)
```

---

## Complete File Structure

```
traffic project/                              ← repo root
│
├── app.py                                    ← CLI: --video, --camera, --dry-run, --show, --dashboard
├── streamlit_app.py                          ← 4-tab cloud demo (Detect, Camera Setup, How It Works, Sample)
├── requirements.txt                          ← Runtime deps only (easyocr, torch, streamlit, etc.)
├── requirements-train.txt                    ← Training deps (roboflow, kaggle)
├── packages.txt                              ← apt deps for cloud deployment (libglib2.0-0t64, libgl1)
├── runtime.txt                               ← Pins Python 3.11 on Streamlit Cloud
├── README.md
├── .gitignore                                ← comprehensive: .pt/.pth/.mp4/DB/venv/cache
├── Dockerfile                                ← python:3.11-slim CPU image; GPU note inside
├── docker-compose.yml                        ← dashboard + pipeline services, shared volumes
├── .dockerignore
│
├── src/configs/
│   ├── pipeline.yaml                         ← device: cuda; vehicle_detector: yolo11s.pt
│   ├── cameras.yaml                          ← stop_line, signal_roi, no_parking_zones per camera
│   └── violations.yaml                       ← thresholds, refresh_interval, min_confirm_frames per violation
│
├── src/
│   ├── components/
│   │   ├── preprocessing/
│   │   │   └── frame_processor.py               ← process_frame() → (frame, FrameQuality)
│   │   ├── detection/
│   │   │   ├── vehicle_detector.py              ← VehicleDetector (YOLO11 wrapper, class whitelist)
│   │   │   └── plate_detector.py               ← PlateDetector + crop-coord translation
│   │   ├── tracking/
│   │   │   ├── tracker.py                      ← Tracker (self-contained IoU — NOT ByteTrack/BYTETracker)
│   │   │   └── track_memory.py                 ← TrackMemory: caches ML results, confirm counters, helmet_bbox
│   │   ├── violations/
│   │   │   ├── classifier.py                   ← route(record) → sets status from per-violation threshold
│   │   │   ├── signal_utils.py                 ← detect_signal_state() → "red"|"green"|"unknown"
│   │   │   ├── triple_riding.py               ← containment rule (min_overlap_ratio, not IoU)
│   │   │   ├── wrong_side.py
│   │   │   ├── stop_line.py
│   │   │   ├── red_light.py
│   │   │   ├── parking.py                     ← key in violations.yaml: illegal_parking
│   │   │   ├── helmet.py                      ← HelmetChecker: full-frame detect + upward-bbox association + confirm counter
│   │   │   └── seatbelt.py                    ← SeatbeltChecker: YOLO11s classify, car/truck/bus only, confirm counter
│   │   ├── ocr/
│   │   │   └── plate_reader.py                 ← PlateReader (EasyOCR wrapper, handles text+conf)
│   │   └── evidence/
│   │       └── generator.py                    ← EvidenceGenerator (annotated image + json sidecar)
│   ├── database/
│   │   ├── schema.py                     ← SQLAlchemy table + init_db()
│   │   └── repository.py               ← insert_violation, query_violations, count_by_*, export_csv
│   ├── analytics/
│   │   └── stats.py
│   └── evaluation/
│       └── metrics.py                   ← classification_metrics, ocr_accuracy, average_precision, FPSTimer
│
├── pipelines/
│   └── video_pipeline.py               ← run() — wires all modules; passes min_confirm_frames to TrackMemory
│
├── dashboard/
│   └── app.py                          ← Streamlit: KPIs, bar chart, trend line, table, image viewer, CSV
│
├── scripts/
│   ├── download_models.py              ← downloads COCO YOLO vehicle weights
│   ├── download_datasets.py           ← pulls helmet/plate datasets (Roboflow/Kaggle)
│   ├── images_to_video.py             ← builds a demo .mp4 from an image folder
│   └── draw_zones.py                  ← interactive OpenCV zone editor → prints cameras.yaml YAML
│
├── models/
│   └── weights/                       ← yolo11s.pt, helmet_yolov8.pt, plate_yolov8.pt, seatbelt_yolov11s.pt (in .gitignore)
│
├── data/
│   ├── raw/                           ← gitignored
│   ├── samples/                       ← test clips (*.mp4 gitignored) + reference images
│   └── seatbelt_crops/
│       ├── seatbelt/
│       └── no_seatbelt/
│
├── artifacts/
│   └── evidence/                      ← runtime output, gitignored
│
├── notebooks/
│   └── 01_train_models_colab.ipynb   ← trains helmet + plate on Colab T4
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_violations.py
│   └── test_ocr.py
│
└── docs/
    ├── Traffic_Violation_Final_Implementation_Plan.md  ← PRIMARY source of truth
    ├── COLAB_GUIDE.md                                  ← Colab training workflow
    ├── DEPLOYMENT.md                                   ← Cloud deploy steps (HF Spaces) + HackerEarth guide
    ├── resources.md                                    ← datasets, models, tools checklist
    └── context.md                                      ← this file
```

---

## Key Data Structures (`src/models.py`)

```python
@dataclass
class Detection:
    class_name: str; confidence: float; bbox: tuple; frame_id: int

@dataclass
class TrackedObject:
    track_id: int; class_name: str; bbox: tuple; confidence: float
    frame_id: int; centroid_history: list[tuple[int, int]]   # last 60 (cx, cy)

@dataclass
class ViolationRecord:
    violation_type: str        # "helmet" | "seatbelt" | "triple_riding" |
                               # "wrong_side" | "stop_line" | "red_light" | "illegal_parking"
    confidence: float; vehicle_id: int; bbox: tuple
    timestamp: str; frame_id: int
    plate_number: str | None; plate_confidence: float | None
    status: str                # "auto_flagged" | "review" | "indeterminate"
    evidence_image_path: str | None; evidence_json_path: str | None
    is_blurry: bool; camera_id: str
```

### TrackState (key fields added recently)

```python
@dataclass
class TrackState:
    # Helmet
    helmet_confirm_count: int = 0    # recheck cycles seeing no_helmet (emit after >= min_helmet_confirm)
    helmet_bbox: Optional[tuple] = None  # tight head bbox (not the motorcycle bbox)
    # Seatbelt
    seatbelt_confirm_count: int = 0  # recheck cycles seeing no_seatbelt
```

---

## Critical Config Notes

| File | Critical key | Value |
|------|-------------|-------|
| `pipeline.yaml` | `models.vehicle_detector` | `models/weights/yolo11s.pt` (YOLO11, not v8) |
| `pipeline.yaml` | `models.seatbelt_classifier` | `models/weights/seatbelt_yolov11s.pt` (auto-downloaded HuggingFace) |
| `pipeline.yaml` | `inference.device` | `cuda` (RTX 3050 Ti confirmed working) |
| `violations.yaml` | parking section key | `illegal_parking` (NOT `parking`) |
| `violations.yaml` | triple riding key | `min_person_overlap_ratio` (NOT `min_person_overlap_iou`) |
| `violations.yaml` | `helmet.min_confirm_frames` | `2` — must see no_helmet in 2 separate recheck cycles |
| `violations.yaml` | `seatbelt.min_confirm_frames` | `2` — same gating for seatbelt |
| `violations.yaml` | `helmet.flag_invalid_helmet` | `true` — also flags improperly-worn helmets |
| `cameras.yaml` | All geometry | Set with `python scripts/draw_zones.py --video <clip>` OR via Camera Setup tab in streamlit_app.py |

---

## Bugs Fixed (history — do not reintroduce)

| Bug | Fix |
|-----|-----|
| `np.trapz` removed in NumPy 2.x | Changed to `np.trapezoid` in `metrics.py` |
| `triple_riding.py` used IoU | Changed to containment (intersection / person-area); IoU missed real triple-riding |
| `tracker.py` used private `BYTETracker` internal API | Replaced with self-contained IoU tracker (no external dep, stable) |
| `helmet.py` cropped top-fraction of motorcycle bbox | COCO motorcycle box excludes rider head; changed to full-frame detect + upward association |
| `plate_reader.py` used PaddleOCR | PaddlePaddle 3.x oneDNN runtime bug on Windows; swapped to EasyOCR |
| `video_pipeline.py` eager ML imports | Made lazy (inside `if not dry_run:`) so `--dry-run` works without torch/easyocr |
| violations.yaml config | `parking` key → `illegal_parking`; `min_person_overlap_iou` → `min_person_overlap_ratio` |
| Expensive ML ran every frame | Implemented `TrackMemory` for helmet/seatbelt/plate caching, synced to tracker |
| DB flooded with `indeterminate` | `TrackMemory` ensures seatbelt `indeterminate` is emitted only once per car |
| Seatbelt false positives on pedestrians | `SeatbeltChecker` now filters to `car/truck/bus` only — `person` explicitly excluded |
| Single blurry frame → false DB violation | `min_confirm_frames=2` in TrackMemory: helmet/seatbelt violations require 2 recheck cycles to agree |
| `helmet_bbox` was a dynamic attribute | Added properly to `TrackState` dataclass |
| `debug.jpg` written to repo root | Friend's debug line removed from `streamlit_app.py` |
| Streamlit app only had 2 tabs (upload + sample) | Upgraded to 4 tabs: Detect, Camera Setup, How It Works, Try Sample |

---

## How to Run

```powershell
cd "c:\Users\rauna\Videos\traffic project"
.\venv\Scripts\activate

# Full pipeline (GPU)
python app.py --video data\samples\test_video.mp4 --show

# Webcam
python app.py --video 0

# RTSP stream
python app.py --video rtsp://192.168.1.10/stream

# Preprocessing-only (no torch/easyocr needed)
python app.py --video data\samples\test_video.mp4 --dry-run

# Streamlit dashboard on http://localhost:8501
python app.py --dashboard

# Cloud demo (4-tab interactive app)
streamlit run streamlit_app.py

# Tests
python -m pytest tests\ -q
```

### Docker

```bash
docker compose build
docker compose up dashboard                          # UI on :8501
docker compose run --rm pipeline python app.py --video data/samples/test_video.mp4
```

---

## Setup From Scratch (New Machine)

```powershell
# 1. Python 3.11 venv (NOT 3.13 — no torch wheels for 3.13)
py -3.11 -m venv venv
.\venv\Scripts\activate

# 2. GPU torch first (MUST precede requirements.txt)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Remaining deps
pip install -r requirements.txt

# 4. Vehicle COCO weights
python scripts/download_models.py

# 5. Place trained weights in models/weights/
#    helmet_yolov8.pt + plate_yolov8.pt (from Colab — see docs/COLAB_GUIDE.md)
#    seatbelt_yolov11s.pt (auto-downloaded from HuggingFace on first run)

# 6. Set up camera zones (run once per camera)
python scripts/draw_zones.py --video data\samples\test_video.mp4
# Paste output into src/configs/cameras.yaml
# OR use the Camera Setup tab in: streamlit run streamlit_app.py

# 7. Run
python app.py --video data\samples\test_video.mp4 --show
```

---

## Training Workflow (Colab)

`scripts/download_datasets.py` pulls datasets → `notebooks/01_train_models_colab.ipynb` trains helmet + plate detectors on Colab T4 (free) → download `best.pt` → place in `models/weights/`.

**Vehicle model is deliberately NOT fine-tuned** — fine-tuning on a vehicle-only set erases the `person` class which triple-riding and helmet association depend on.

See `docs/COLAB_GUIDE.md` for step-by-step.

---

## Known Limitations

- **Seatbelt**: YOLO11s classifier from HuggingFace (RISEF/yolov11s-seatbelt) is working but marks `indeterminate` when windshield crop is too small. Always requires 2 recheck cycles to confirm.
- **Auto-rickshaw**: COCO has no such class; falls under `motorcycle` / `car`
- **Rain handling**: classical filter only (median blur + unsharp); not a deraining network
- **Signal detection**: HSV-based; may fail under overexposure or night wash-out
- **Parking dwell timer**: resets on track ID switch under occlusion
- **Streamlit cloud app**: geometry violations (stop-line, parking) require camera zones to be configured in the Camera Setup tab first; wrong-side and red-light only run in the full local pipeline

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vehicle/person detection | YOLO11s (Ultralytics) — pretrained COCO |
| Plate detection | YOLO — Colab-trained, class `License_Plate` |
| Helmet detection | YOLO — Colab-trained, full-frame + association |
| Seatbelt | YOLO11s classify — `RISEF/yolov11s-seatbelt` (HuggingFace, auto-downloaded) |
| Tracking | Self-contained greedy IoU tracker |
| OCR | EasyOCR (GPU via torch) |
| Preprocessing | OpenCV (CLAHE, Laplacian, median blur) |
| Database | SQLite via SQLAlchemy |
| Dashboard | Streamlit (local analytics: `dashboard/app.py`) |
| Cloud Demo | Streamlit 4-tab app (`streamlit_app.py`) — Detect, Camera Setup, How It Works, Try Sample |
| Optimization | `TrackMemory` — per-track caching, confirm counters, occlusion recheck |
| Deployment | Docker + docker-compose |
| Evaluation | scikit-learn + custom mAP (np.trapezoid) |
| Tests | pytest (36/36 passing) |

---

## What Still Needs to Be Done

| Item | Priority |
|------|----------|
| Plate model mAP metrics (screenshot for report) | High |
| Real Indian traffic video with riders + plates for demo | High |
| `notebooks/03_evaluation.ipynb` | Medium |
| Seatbelt dataset collection (optional — HF model is working) | Low |
| Auto-rickshaw fine-tuning (IDD dataset) | Low |
