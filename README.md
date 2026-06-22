# Traffic Violation Detection System

Automated photo/video identification and classification of traffic violations using computer vision.

## Violations Detected

| Violation | Approach |
|-----------|----------|
| Helmet non-compliance | Full-frame YOLO detector → head associated to motorcycle track via upward-expanded bbox |
| Seatbelt non-compliance | YOLO11s classify on windshield crop (car/truck/bus only) — auto-downloaded from HuggingFace |
| Triple riding | Person-containment rule on motorcycle bbox (≥ 3 persons → violation) |
| Wrong-side driving | Direction vector + centroid track history |
| Stop-line violation | Virtual line + signal state (HSV) |
| Red-light violation | Signal ROI + vehicle position |
| Illegal parking | No-parking polygon + dwell timer (3 min) |

## Models

| Model | File | How to get it |
|-------|------|---------------|
| Vehicle + person + road users | `models/weights/yolo11s.pt` | `python scripts/download_models.py` (auto) |
| Helmet / no-helmet | `models/weights/helmet_yolov8.pt` | Train on Colab → [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) |
| License plate | `models/weights/plate_yolov8.pt` | Train on Colab → [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) |
| Seatbelt classifier | `models/weights/seatbelt_yolov11s.pt` | **Auto-downloaded from HuggingFace** on first run |

Download the training datasets locally with `python scripts/download_datasets.py --all`
(needs a free Roboflow or Kaggle key), or let the Colab notebook pull them.

## Setup

> **Use Python 3.11.** `torch` does not publish wheels for Python 3.13. The Docker image pins 3.11 for you.

```bash
# 1. Create virtual environment (Python 3.11)
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install GPU torch first (Windows / Linux with CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Download pretrained model weights
python scripts/download_models.py
# The seatbelt model downloads automatically from HuggingFace on first pipeline run.

# 5. Set up camera zones (run once per camera)
python scripts/draw_zones.py --video data/samples/test_video.mp4
# Copy the printed YAML into src/configs/cameras.yaml
# OR use the Camera Setup tab in the Streamlit app (see Cloud Demo below)

# 6. Configure pipeline
# Edit src/configs/pipeline.yaml  → set input source, model paths, device (cuda/cpu)
# Edit src/configs/violations.yaml → adjust thresholds / min_confirm_frames if needed
```

## Running the Pipeline

```bash
# Process a video file
python app.py --video data/samples/test_video.mp4

# Webcam
python app.py --video 0

# RTSP stream
python app.py --video rtsp://192.168.1.10/stream

# Show annotated output window
python app.py --video data/samples/test_video.mp4 --show

# Dry run (preprocessing only, no ML models loaded)
python app.py --video data/samples/test_video.mp4 --dry-run
```

## Dashboard

```bash
python app.py --dashboard
# Opens local Streamlit analytics dashboard at http://localhost:8501
```

## Cloud Demo

An interactive Streamlit app with 4 tabs:

| Tab | What it does |
|-----|-------------|
| 🎯 Detect Violations | Upload image or video → runs helmet, seatbelt, triple riding, plates. Runs stop-line + parking checks too if camera zones are configured. |
| 🗺️ Camera Setup | Define stop line, signal ROI, and no-parking zones on a reference frame using coordinate inputs with live preview. |
| ℹ️ How It Works | Full explainer for all 8 pipeline stages and all 7 violation types. |
| 🖼️ Try a Sample | Run detection on pre-loaded footage from `data/samples/`. |

```bash
# Run the cloud demo locally
streamlit run streamlit_app.py
```

For deployment to Streamlit Community Cloud or Hugging Face Spaces, see **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

## Deployment (Docker)

Reproducible, runs anywhere — no local Python/CUDA setup needed. Requires Docker Desktop.

```bash
# Build the image (pins Python 3.11 + all deps)
docker compose build

# Launch the analytics dashboard  ->  http://localhost:8501
docker compose up dashboard

# Process a video into the shared database (dashboard updates live)
docker compose run --rm pipeline python app.py --video data/samples/test_video.mp4
```

The `artifacts/`, `models/`, and `data/` folders are mounted as volumes, so trained
weights, the violations database, and evidence images persist across runs and are
shared between the pipeline and dashboard containers. For GPU inference, see the
note at the top of the [Dockerfile](Dockerfile).

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
traffic_violation_system/
├── src/configs/           # All tunable parameters (YAML — no hardcoded values)
│   ├── pipeline.yaml  #   device, model paths, FPS target
│   ├── cameras.yaml   #   stop line, signal ROI, parking zones per camera
│   └── violations.yaml#   thresholds, refresh_interval, min_confirm_frames
├── src/
│   ├── preprocessing/ # CLAHE, blur detection, rain filter
│   ├── detection/     # YOLO11 vehicle + YOLOv8 plate/helmet detectors
│   ├── tracking/      # Self-contained IoU tracker + TrackMemory caching
│   ├── violations/    # One file per violation type + classifier.py
│   ├── ocr/           # EasyOCR plate reader
│   ├── evidence/      # Annotated image + JSON saver
│   ├── database/      # SQLite schema + repository
│   ├── analytics/     # Aggregation queries for dashboard
│   └── evaluation/    # Metrics: mAP, F1, OCR accuracy, FPS
├── pipelines/         # Main frame loop (orchestrator)
├── dashboard/         # Local Streamlit analytics dashboard
├── scripts/           # Setup tools (download models, draw zones)
├── models/weights/    # .pt files (gitignored — download separately)
├── data/samples/      # Short test clips and reference images
├── artifacts/evidence/# Saved violation images + JSON (runtime, gitignored)
├── tests/             # Pytest unit tests (36/36 passing)
└── notebooks/         # Colab training notebook
```

## Multi-Frame Confirmation

Helmet and seatbelt violations require **2 separate recheck cycles** to agree before being logged. This prevents a single blurry or shadowed frame from creating a false positive in the database.

- Configurable via `min_confirm_frames` in `src/configs/violations.yaml`
- A clean detection (no violation found) resets the counter
- Single-image mode (cloud demo) emits immediately (no tracker, no confirm count)

## Known Limitations

- **Seatbelt**: `indeterminate` when windshield crop is too small. Always requires 2 recheck cycles. Pedestrians are explicitly excluded from the check.
- **Auto-rickshaw**: Not in COCO — pretrained model uses `motorcycle` as a proxy.
- **Rain handling**: Classical median blur + sharpen only.
- **Signal detection**: HSV-based; may misclassify under overexposed or nighttime conditions.
- **Geometry violations in cloud demo**: Stop-line and parking require camera zones to be set in the Camera Setup tab. Wrong-side and red-light run only in the full local pipeline.
- **Multiple cameras**: Supported via `--camera <id>` + separate entries in `cameras.yaml`.

## Evaluation

See `notebooks/03_evaluation.ipynb`.

Metrics computed:
- mAP@0.5 — vehicle and plate detectors
- Precision / Recall / F1 — per violation type
- OCR exact-match accuracy
- FPS throughput
