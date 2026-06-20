# Resource Collection Guide

Everything you need to gather before the pipeline can run end-to-end.
Organized by priority — collect in this order.

---

## PRIORITY 1 — Model Weights (Pipeline Won't Start Without These)

### 1.1 YOLO11s — Vehicle Detection (Auto-downloaded)
- **What**: Pretrained COCO weights. Detects car, truck, bus, motorcycle, person.
- **How to get**: Runs automatically on first use via Ultralytics.
  ```bash
  python scripts/download_models.py
  ```
- **File to place**: `models/weights/yolo11s.pt`
- **Size**: ~18 MB
- **Cost**: Free
- **Status**: ✅ Auto-downloaded

---

### 1.2 Helmet Detection Model — Fine-tuned YOLOv8
- **What**: A YOLOv8 classifier that distinguishes helmet vs no-helmet on cropped head regions.
- **Option A — Use a pretrained public model (Recommended)**
  - Search on [Roboflow Universe](https://universe.roboflow.com): search `"helmet detection"`
  - Pick a model with ≥ 1000 images, Indian road context preferred
  - Download as YOLOv8 format → get the `.pt` weights
  - Free with Roboflow account
- **Option B — Fine-tune yourself**
  - Dataset: [Helmet Detection Dataset on Roboflow](https://universe.roboflow.com/roboflow-universe-projects/hard-hat-universe) or search Kaggle for "helmet motorcycle India"
  - Fine-tune `yolov8n-cls.pt` on it using Ultralytics (see notebook)
- **File to place**: `models/weights/helmet_yolov8.pt`
- **Status**: ❌ Must collect

---

### 1.3 Indian License Plate Detector — Fine-tuned YOLOv8
- **What**: YOLOv8 fine-tuned to localize Indian license plates in traffic footage.
- **Option A — Pretrained public model (Recommended)**
  - [Roboflow Universe — License Plate Recognition](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)
  - Search for "Indian number plate detection" on Roboflow
  - Download YOLOv8 format `.pt` weights
- **Option B — Kaggle datasets for fine-tuning**
  - [Indian Number Plate Dataset — Kaggle](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset)
  - [Vehicle Registration Plates — Kaggle](https://www.kaggle.com/datasets/nickyazdani/license-plate-text-detection)
- **File to place**: `models/weights/plate_yolov8.pt`
- **Status**: ❌ Must collect

---

## PRIORITY 2 — Datasets (Needed for Training & Evaluation)

### 2.1 Seatbelt Classifier Dataset — MUST COLLECT MANUALLY
- **What**: ~300–500 images of car windshields from roadside CCTV angle.
  - Label: `seatbelt` (belt visible) / `no_seatbelt` (belt not visible)
- **Why manual**: No good public dataset exists for this exact camera angle.
- **Sources to try**:
  - Google Images / Bing Images — search "car seatbelt roadside CCTV", "driver seatbelt dashboard view"
  - YouTube traffic enforcement videos — screenshot frames
  - Your own test camera footage
- **Minimum viable**: 150 images per class (300 total) for a working binary classifier
- **Where to place**:
  ```
  data/seatbelt_crops/seatbelt/         ← ~150+ images
  data/seatbelt_crops/no_seatbelt/      ← ~150+ images
  ```
- **Then run**: `notebooks/02_seatbelt_training.ipynb`
- **Status**: ❌ Must collect manually

---

### 2.2 Test Video Footage — For Running the Pipeline
- **What**: Traffic footage from a road intersection showing vehicles.
- **Sources (Free)**:
  - [AI City Challenge Dataset](https://www.aicitychallenge.org/) — registration required, free
  - [MIO-TCD Dataset](https://lts2.epfl.ch/datasets/mio-tcd/) — traffic classification dataset
  - YouTube — search "India traffic intersection CCTV footage", download with `yt-dlp`
    ```bash
    pip install yt-dlp
    yt-dlp -f mp4 "<youtube-url>" -o "data/samples/test_video.mp4"
    ```
  - Your own phone/dashcam footage of a junction
- **Where to place**: `data/samples/test_video.mp4`
- **Status**: ❌ Must collect

---

### 2.3 Evaluation Ground Truth — For Running Metrics
- **What**: A manually-labelled set of 100–200 frames with known violations.
- **Format needed**: Simple CSV or JSON:
  ```
  frame_id, violation_type, bbox_x1, bbox_y1, bbox_x2, bbox_y2
  ```
- **How to create**:
  - Pick 100–200 frames from your test video
  - Label them using [Label Studio](https://labelstud.io/) (free, runs locally) or [CVAT](https://cvat.ai/) (free)
  - Export in YOLO or COCO format
- **Why needed**: Without this, you cannot compute Precision/Recall/F1/mAP from `notebooks/03_evaluation.ipynb`
- **Status**: ❌ Must create (even 100 frames is enough)

---

### 2.4 Indian Road Dataset — For Fine-tuning Vehicle Detector (Optional but Recommended)
- **What**: Indian road footage to handle auto-rickshaws and local traffic density.
- **Sources**:
  - [IDD — India Driving Dataset](https://idd.insaan.iiit.ac.in/) — free, registration required
  - [DriveIndia Dataset](https://github.com/seyrankhademi/ResNet_Traffic_Sign) — dashcam clips
  - [IIIT Hyderabad Road Dataset](http://cvit.iiit.ac.in/research/projects/cvit-projects/ms-coco-like-dataset-for-indian-roads)
- **Why**: COCO-pretrained YOLOv8 will miss auto-rickshaws. Fine-tune for better accuracy.
- **Status**: ⚠️ Optional — use if pretrained accuracy is poor on your footage

---

## PRIORITY 3 — Tools & Accounts (Free, Setup Once)

### 3.1 Roboflow Account
- **What**: Platform to browse pretrained models, download datasets, manage labels.
- **Why**: Best source for helmet and plate detection models in YOLOv8 format.
- **URL**: https://roboflow.com
- **Cost**: Free tier is sufficient
- **Status**: ❌ Create account if you don't have one

---

### 3.2 Kaggle Account
- **What**: Dataset downloads (Indian plates, helmet datasets).
- **URL**: https://kaggle.com
- **Cost**: Free
- **Status**: ❌ Create account if you don't have one

---

### 3.3 Label Studio (Local Labelling Tool)
- **What**: Open source annotation tool — use to label seatbelt crops and evaluation frames.
- **Install**:
  ```bash
  pip install label-studio
  label-studio start
  ```
- **URL**: http://localhost:8080 (opens in browser)
- **Cost**: Free, runs 100% locally
- **Status**: ❌ Install when ready to label

---

## PRIORITY 4 — No API Keys Required

> This project deliberately uses **no paid APIs or cloud services**.

| Component | Technology | Cost |
|-----------|-----------|------|
| Vehicle detection | YOLO11s pretrained (COCO) | Free |
| OCR | EasyOCR (runs locally via torch) | Free |
| Tracking | Self-contained IoU tracker (`src/tracking/tracker.py`) | Free |
| Database | SQLite (local file) | Free |
| Dashboard | Streamlit (local) | Free |
| Seatbelt model | Trained locally in notebook | Free |

No OpenAI, no Google Vision API, no AWS, no Azure — everything runs on your machine.

---

## Summary Checklist

| Resource | Source | Priority | Status |
|----------|--------|----------|--------|
| `yolo11s.pt` (vehicle) | Auto-downloaded by Ultralytics | P1 | ✅ Done |
| `helmet_yolov8.pt` | **Trained on Colab T4** | P1 | ✅ Done |
| `plate_yolov8.pt` | **Trained on Colab T4** | P1 | ✅ Done |
| EasyOCR (replaces PaddleOCR) | pip install easyocr | P1 | ✅ Done |
| Docker setup | Dockerfile + docker-compose.yml | P1 | ✅ Done |
| Test video (`.mp4`) | `data/samples/indian_traffic_test_video.mp4` | P2 | ✅ Done |
| Camera zone config | `configs/cameras.yaml` (set via `draw_zones.py`) | P2 | ✅ Done |
| Seatbelt crop images | Manual collection (~300) | P2 | ❌ Optional |
| Plate model mAP metrics | Capture from Colab training run | P2 | ❌ Still needed for report |
| Evaluation labels (100–200 frames) | Label Studio / CVAT | P2 | ❌ For report |
| Indian road dataset (IDD) | idd.insaan.iiit.ac.in | P3 optional | ⚠️ Only if accuracy is poor |
| Roboflow account | roboflow.com | P3 | ✅ Used for training |
| Kaggle account | kaggle.com | P3 | ✅ Used for training |

---

## Minimum to Get the Pipeline Running (Demo-Ready)

✅ **Models are already trained and in `models/weights/`.** The minimum you still need:

1. **A real Indian traffic video** → `data/samples/test_video.mp4`
   - Use `scripts/images_to_video.py` on helmet dataset images as a stopgap
2. **Camera zone config** → run `python scripts/draw_zones.py --video <your_clip>` and paste output into `configs/cameras.yaml`
3. **Plate model mAP** → go back to the Colab training run, cell 5 prints metrics — screenshot for the report

Seatbelt shows as `indeterminate` — correct honest fallback, not an error.
