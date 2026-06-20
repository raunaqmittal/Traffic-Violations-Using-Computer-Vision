# Google Colab Training Guide

You have a **4 GB GTX 1650** — perfect for *running* the system at inference (~25–30 FPS with `yolo11s`), but training detectors needs more VRAM. Train on Colab's free **T4 (16 GB)**, then download the weights and run everything locally.

## What gets trained (and what doesn't)

| Model | Trained? | Where |
|---|---|---|
| Vehicle + person + road users | **No** — pretrained COCO YOLO already detects car/bus/truck/motorcycle/person accurately | auto-downloaded locally |
| Helmet / no-helmet | **Yes** | Colab |
| License plate | **Yes** | Colab |
| Seatbelt (binary CNN) | Optional | local/Colab, small manual set |

We deliberately **do not** fine-tune the vehicle model on a vehicle-only dataset — doing so erases the `person` class, which the triple-riding and helmet logic depend on.

## Step-by-step

1. **Get a free Roboflow API key** — https://app.roboflow.com → Settings → API. (Or a Kaggle token: kaggle.com → Account → Create API Token → `kaggle.json`.)
2. **Open the notebook in Colab:** upload `notebooks/01_train_models_colab.ipynb` to https://colab.research.google.com.
3. **Enable GPU:** `Runtime → Change runtime type → T4 GPU`.
4. **Run the cells top to bottom:**
   - Cell 1 installs deps and confirms the T4 is active.
   - Cell 2a downloads the helmet + plate datasets (paste your Roboflow key). Cell 2b is the Kaggle alternative.
   - Cells 3–4 train the two detectors (~1–2 hrs total on T4).
   - Cell 5 prints mAP / Precision / Recall — **screenshot this for your report.**
   - Cell 6 downloads `helmet_yolov8.pt` and `plate_yolov8.pt`.
5. **Place the weights locally:**
   ```
   models/weights/helmet_yolov8.pt
   models/weights/plate_yolov8.pt
   ```
   The vehicle model downloads itself on first run via `python scripts/download_models.py`.

## Picking a better dataset

To swap in a different (larger / higher-quality) dataset:
- Browse https://universe.roboflow.com → search `"indian helmet"` or `"indian number plate"`.
- Prefer one with the most images, a permissive license (CC BY / MIT), and Indian road context.
- Open it → `Download → YOLOv8` → copy the `workspace`, `project`, `version` strings into the notebook's Cell 2a (and into `scripts/download_datasets.py` if you want the local downloader to match).

## Running locally after training

```powershell
python -m venv venv; .\venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py            # vehicle (COCO) weights
# put the two trained .pt files in models/weights/
# set device: "cuda" in configs/pipeline.yaml to use the GTX 1650
python app.py --video data/samples/test_video.mp4 --show
```
