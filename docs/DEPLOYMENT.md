# Deployment & Hackathon Submission Guide

This covers getting a **public Demo Link** for the HackerEarth submission, plus the
exact text/files for every required field.

The demo entry point is [`streamlit_app.py`](../streamlit_app.py) — a CPU-friendly
interactive app: upload a traffic image → it detects road users, flags helmet /
triple-riding violations, and reads number plates, returning annotated evidence.

---

## Prerequisite: commit the model weights

The cloud app needs the trained weights in the repo (they are small — ~19 MB each,
well under GitHub's 100 MB limit). `.gitignore` already allows `*.pt`. Make sure
these are committed:

```
models/weights/yolo11s.pt
models/weights/helmet_yolov8.pt
models/weights/plate_yolov8.pt
```

Also add a sample image or two to `samples/` (e.g. a helmet-dataset test image with
a no-helmet rider) so judges see a violation fire immediately.

---

## Option A — Streamlit Community Cloud (easiest, repo already on GitHub)

1. Go to **https://share.streamlit.io** → sign in with GitHub.
2. **New app** → pick your repo, branch `feature/training-deploy-pipeline` (or `main`
   after you merge), main file **`streamlit_app.py`**.
3. **Deploy.** First build takes a few minutes (installs torch CPU + EasyOCR).
4. You get a public URL like `https://<name>.streamlit.app` → **this is your Demo Link.**

`requirements.txt` and `packages.txt` (system libs) are already set up for this.

> Free tier RAM is ~1 GB. Three small YOLO models + EasyOCR usually fit. If it OOMs,
> use Option B (16 GB).

## Option B — Hugging Face Spaces (more RAM, ML-friendly)

1. https://huggingface.co/new-space → SDK **Streamlit**, hardware **CPU basic** (free, 16 GB).
2. Push this repo to the Space (or upload). Weights >10 MB go via git-lfs:
   ```bash
   git lfs install
   git lfs track "*.pt"
   ```
3. In the Space's `README.md` metadata, set the entry point (the project already has an
   `app.py` CLI, so you MUST point Streamlit at the demo file):
   ```yaml
   ---
   title: Traffic Violation Detection
   emoji: 🚦
   sdk: streamlit
   app_file: streamlit_app.py
   ---
   ```
4. The Space builds and serves a public URL → **Demo Link.**

---

## Filling the HackerEarth submission form

| Field | What to put |
|---|---|
| **Snapshots** | Screenshots of the running demo: the annotated image with violation boxes, the results table, the dashboard. (JPG/PNG ≤ 3 MB each.) |
| **Video URL** | A 2–3 min screen recording: upload an image → show violations flagged + plate read → show the analytics dashboard. Upload to YouTube (unlisted) and paste the link. |
| **Presentation** | Pitch deck (problem → approach → architecture → results/metrics → demo). Export as PDF. |
| **Demo Link** | The Streamlit/HF Spaces URL from Option A or B. |
| **Repository URL** | Your GitHub repo URL. |
| **Source Code** | A zip ≤ 50 MB — see the command below (exclude venv/weights/artifacts). |
| **Instructions to Run** | Paste the block below. |

### Source-code zip (≤ 50 MB, excludes heavy folders)

```bash
git archive --format=zip -o submission_source.zip HEAD \
  ":(exclude)models/weights/*.pt" ":(exclude)samples/*"
```
(The weights are reachable via the Repository URL and the live Demo Link, so they
don't need to be in the 50 MB source zip.)

---

## Instructions to Run (paste into the form)

**Live demo:** open the Demo Link, upload a traffic image (or pick a sample), and the
system returns annotated evidence with detected violations and plate numbers.

**Run locally (full system, GPU optional):**

1. Requires **Python 3.11** (paddle/torch have no 3.13 wheels).
   ```
   py -3.11 -m venv venv && venv\Scripts\activate
   ```
2. Install dependencies (GPU torch shown; drop the index-url for CPU):
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
3. Get the vehicle weights (helmet/plate weights are already in `models/weights/`):
   ```
   python scripts/download_models.py
   ```
4. **Interactive demo:**  `streamlit run streamlit_app.py`  → http://localhost:8501
5. **Full video pipeline:**  `python app.py --video data/samples/test_video.mp4 --show`
6. **Analytics dashboard:**  `python app.py --dashboard`
7. **Docker (no setup):**  `docker compose up dashboard`

Tests: `python -m pytest tests/ -q`
