"""
Download the training datasets for the Traffic Violation Detection System.

The system uses THREE detectors, but only TWO need custom training:

  1. Vehicle + person + road users  -> pretrained COCO YOLO (NO dataset needed,
     auto-downloaded by Ultralytics). Detecting `person` is required for the
     triple-riding and helmet-association logic, so we deliberately do NOT
     fine-tune the vehicle model on a vehicle-only set (that would erase the
     `person` class).
  2. Helmet / no-helmet (+ rider, plate context)  -> trained on an Indian
     helmet dataset.
  3. License-plate localization  -> trained on an Indian number-plate dataset.

This script pulls (2) and (3) from Roboflow Universe and/or Kaggle into
data/ in YOLOv8 format, ready for notebooks/01_train_models_colab.ipynb.

Credentials (free accounts):
  - Roboflow: set ROBOFLOW_API_KEY  (https://app.roboflow.com -> Settings -> API)
  - Kaggle:   place kaggle.json in ~/.kaggle/ (https://kaggle.com -> Account -> Create API Token)

Usage:
  python scripts/download_datasets.py --helmet --plate          # both, via Roboflow
  python scripts/download_datasets.py --helmet --source kaggle  # helmet from Kaggle
  python scripts/download_datasets.py --all

Run with no flags to print this guidance.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Recommended dataset sources. Override any of these with CLI flags if you find
# a better one on Roboflow Universe (search "indian helmet" / "indian number
# plate", pick one with the most images + a permissive license, then copy its
# workspace / project / version from the Roboflow "Download -> YOLOv8" panel).
# ---------------------------------------------------------------------------
ROBOFLOW_HELMET = {
    "workspace": "vehicles-dataset-zoqwx",
    "project": "motorcycle-helmet-detection-dataset-axtx6",
    "version": 1,
}
ROBOFLOW_PLATE = {
    # Large Indian-plate set; swap for any plate project you prefer.
    "workspace": "roboflow-universe-projects",
    "project": "license-plate-recognition-rxg4e",
    "version": 11,
}

KAGGLE_HELMET = "aryanvaid13/indian-helmet-detection-dataset"
KAGGLE_PLATE = "andrewmvd/car-plate-detection"


def _roboflow_download(cfg: dict, out_name: str) -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ERROR: ROBOFLOW_API_KEY not set.\n"
            "  PowerShell:  $env:ROBOFLOW_API_KEY='your_key'\n"
            "  bash:        export ROBOFLOW_API_KEY=your_key"
        )
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("ERROR: pip install roboflow")

    target = DATA_DIR / out_name
    target.mkdir(parents=True, exist_ok=True)
    print(f"[roboflow] {cfg['workspace']}/{cfg['project']}:v{cfg['version']} -> {target}")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(cfg["workspace"]).project(cfg["project"])
    project.version(cfg["version"]).download("yolov8", location=str(target))
    print(f"  done. data.yaml at: {target / 'data.yaml'}")


def _kaggle_download(slug: str, out_name: str) -> None:
    try:
        import kaggle  # noqa: F401  (import triggers auth)
    except (ImportError, OSError) as exc:
        sys.exit(f"ERROR: kaggle not ready ({exc}). pip install kaggle and place kaggle.json in ~/.kaggle/")

    target = DATA_DIR / out_name
    target.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] {slug} -> {target}")
    os.system(f'kaggle datasets download -d {slug} -p "{target}" --unzip')
    # Some Kaggle archives don't auto-unzip cleanly; do it defensively.
    for zf in target.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(target)
        zf.unlink()
    print(f"  done. inspect {target} for data.yaml / images/ labels/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--helmet", action="store_true", help="download helmet dataset")
    ap.add_argument("--plate", action="store_true", help="download license-plate dataset")
    ap.add_argument("--all", action="store_true", help="download both")
    ap.add_argument("--source", choices=["roboflow", "kaggle"], default="roboflow")
    args = ap.parse_args()

    want_helmet = args.helmet or args.all
    want_plate = args.plate or args.all

    if not (want_helmet or want_plate):
        print(__doc__)
        return

    if want_helmet:
        if args.source == "roboflow":
            _roboflow_download(ROBOFLOW_HELMET, "helmet_dataset")
        else:
            _kaggle_download(KAGGLE_HELMET, "helmet_dataset")

    if want_plate:
        if args.source == "roboflow":
            _roboflow_download(ROBOFLOW_PLATE, "plate_dataset")
        else:
            _kaggle_download(KAGGLE_PLATE, "plate_dataset")

    print("\nNext: open notebooks/01_train_models_colab.ipynb (Colab T4) to train.")


if __name__ == "__main__":
    main()
