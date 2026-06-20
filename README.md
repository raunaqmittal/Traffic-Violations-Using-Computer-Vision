# Traffic Violation Detection System

Automated photo/video identification and classification of traffic violations using computer vision.

## Violations Detected

| Violation | Approach |
|-----------|----------|
| Helmet non-compliance | YOLOv8 classifier on head crop |
| Seatbelt non-compliance | Binary CNN on windshield crop |
| Triple riding | Person-count rule on motorcycle bbox |
| Wrong-side driving | Direction vector + track history |
| Stop-line violation | Virtual line + signal state (HSV) |
| Red-light violation | Signal ROI + vehicle position |
| Illegal parking | No-parking polygon + dwell timer |

## Models

Three detectors; only **two need training** (the vehicle/person detector is pretrained COCO YOLO):

| Model | File | How to get it |
|-------|------|---------------|
| Vehicle + person + road users | `models/weights/yolo11s.pt` | `python scripts/download_models.py` (auto) |
| Helmet / no-helmet | `models/weights/helmet_yolov8.pt` | Train on Colab → [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) |
| License plate | `models/weights/plate_yolov8.pt` | Train on Colab → [docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md) |

Download the training datasets locally with `python scripts/download_datasets.py --all`
(needs a free Roboflow or Kaggle key), or let the Colab notebook pull them.

## Setup

> **Use Python 3.11.** `paddlepaddle`/`paddleocr` and `torch` do not publish wheels
> for Python 3.13, so a bare 3.13 install fails. The Docker image pins 3.11 for you.

```bash
# 1. Create virtual environment (Python 3.11)
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download pretrained model weights
python scripts/download_models.py

# 4. Set up camera zones (run once per camera)
python scripts/draw_zones.py --video data/samples/test_video.mp4
# Copy the printed YAML into configs/cameras.yaml

# 5. Configure pipeline
# Edit configs/pipeline.yaml  → set input source, model paths, device
# Edit configs/violations.yaml → adjust thresholds if needed
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

# Dry run (preprocessing only, no detection)
python app.py --video data/samples/test_video.mp4 --dry-run
```

## Dashboard

```bash
python app.py --dashboard
# Opens Streamlit dashboard at http://localhost:8501
```

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

## Cloud Demo & Submission

A fully interactive, CPU-friendly Streamlit web app is included for easy deployment to cloud platforms like Hugging Face Spaces or Streamlit Cloud. 

```bash
# Run the cloud demo locally
streamlit run streamlit_app.py
```

For detailed instructions on deploying to the cloud and submitting the project to HackerEarth, please see **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Project Structure

```
traffic_violation_system/
├── configs/           # All tunable parameters (YAML — no hardcoded values)
├── src/
│   ├── preprocessing/ # CLAHE, blur detection, rain filter
│   ├── detection/     # YOLO11 vehicle + YOLOv8 plate/helmet detectors
│   ├── tracking/      # Self-contained IoU tracker + memory caching
│   ├── violations/    # One file per violation type + classifier.py
│   ├── ocr/           # EasyOCR plate reader
│   ├── evidence/      # Annotated image + JSON saver
│   ├── database/      # SQLite schema + repository
│   ├── analytics/     # Aggregation queries for dashboard
│   └── evaluation/    # Metrics: mAP, F1, OCR accuracy, FPS
├── pipelines/         # Main frame loop (orchestrator)
├── dashboard/         # Streamlit analytics dashboard
├── scripts/           # Setup tools (download models, draw zones)
├── models/weights/    # .pt files (gitignored — download separately)
├── data/samples/      # Short test clips
├── artifacts/evidence/# Saved violation images + JSON (runtime)
├── tests/             # Pytest unit tests
└── notebooks/         # Training (seatbelt CNN) + evaluation
```

## Known Limitations

- **Seatbelt**: Marked `indeterminate` when windshield crop is too small or model is not trained. A small labelled dataset (~300 windshield crops) is required to train the binary CNN.
- **Auto-rickshaw**: Not in COCO — pretrained model uses `motorcycle` as a proxy. Fine-tune on IDD dataset for improved accuracy.
- **Rain handling**: Classical median blur + sharpen only. Does not fully restore heavily rain-obscured frames.
- **Signal detection**: HSV-based; may misclassify under overexposed or nighttime conditions. Combine with vehicle-position heuristic for robustness.
- **Multiple cameras**: Supported via `--camera <id>` argument + separate entries in `cameras.yaml`. Concurrent streams require running multiple processes.

## Seatbelt Model Training

See `notebooks/02_seatbelt_training.ipynb`.

Collect windshield crops into:
```
data/seatbelt_crops/seatbelt/       # images with seatbelt worn
data/seatbelt_crops/no_seatbelt/    # images without seatbelt
```
Then run the notebook. Output: `models/seatbelt_classifier/seatbelt_cnn.pth`

## Evaluation

See `notebooks/03_evaluation.ipynb`.

Metrics computed:
- mAP@0.5 — vehicle and plate detectors
- Precision / Recall / F1 — per violation type
- OCR exact-match accuracy
- FPS throughput
