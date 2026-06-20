"""
Download the pretrained vehicle/person detector (COCO YOLO).

This is the ONLY model that auto-downloads — it already detects
car, truck, bus, motorcycle and person, which is everything the
vehicle stage and the triple-riding/helmet association logic need.

The helmet and plate detectors are trained separately on Colab
(see notebooks/01_train_models_colab.ipynb) and dropped into
models/weights/ manually as helmet_yolov8.pt and plate_yolov8.pt.

Usage:
  python scripts/download_models.py            # default: yolov8m
  python scripts/download_models.py --model yolo11s   # lighter, great for GTX 1650
"""

import argparse
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "weights"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="yolov8m",
        help="ultralytics model name (e.g. yolov8m, yolov8s, yolo11s)",
    )
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    target = MODELS_DIR / name

    if target.exists():
        print(f"Already present: {target}")
        return

    from ultralytics import YOLO

    # Instantiating triggers Ultralytics' auto-download; .ckpt_path is the cached file.
    print(f"Downloading {name} via Ultralytics ...")
    model = YOLO(name)
    src = Path(getattr(model, "ckpt_path", "") or name)
    if src.exists() and src.resolve() != target.resolve():
        shutil.copy(src, target)
    print(f"Saved vehicle detector -> {target}")
    print("Reminder: place helmet_yolov8.pt and plate_yolov8.pt in models/weights/ "
          "after training on Colab (notebooks/01_train_models_colab.ipynb).")


if __name__ == "__main__":
    main()
