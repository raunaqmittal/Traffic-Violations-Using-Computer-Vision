"""
Build a pipeline-ready video from a folder of images.

Why: the per-image violations (helmet, triple-riding, license-plate OCR) fire on
single frames, so a slideshow video of REAL Indian road images (e.g. your
helmet_dataset / plate_dataset) is a better violation demo than clean stock
footage where nobody breaks any rules. Each image is held for several frames so
ByteTrack assigns stable track IDs and the detectors have time to fire.

Usage:
  python scripts/images_to_video.py --input data/helmet_dataset/test/images \
                                    --output data/samples/test_video.mp4
  python scripts/images_to_video.py --input <folder> --hold 8 --fps 10 --width 1280
"""

import argparse
import sys
from pathlib import Path

import cv2

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of images")
    ap.add_argument("--output", default="data/samples/test_video.mp4", help="output .mp4 path")
    ap.add_argument("--hold", type=int, default=8, help="frames to hold each image (lets tracker stabilise)")
    ap.add_argument("--fps", type=int, default=10, help="output video fps")
    ap.add_argument("--width", type=int, default=1280, help="resize width (keeps aspect ratio); 0 = keep original")
    ap.add_argument("--limit", type=int, default=0, help="max images to include (0 = all)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    if not in_dir.is_dir():
        sys.exit(f"ERROR: not a folder: {in_dir}")

    images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in _EXTS)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        sys.exit(f"ERROR: no images found in {in_dir}")

    # Determine output frame size from the first image.
    first = cv2.imread(str(images[0]))
    if first is None:
        sys.exit(f"ERROR: cannot read {images[0]}")
    h, w = first.shape[:2]
    if args.width > 0:
        scale = args.width / w
        out_w, out_h = args.width, int(h * scale)
    else:
        out_w, out_h = w, h

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))

    written = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != (out_w, out_h):
            img = cv2.resize(img, (out_w, out_h))
        for _ in range(args.hold):
            writer.write(img)
            written += 1

    writer.release()
    print(f"Wrote {out_path}  |  {len(images)} images x {args.hold} frames = {written} frames @ {args.fps}fps")
    print(f"Run it:  python app.py --video {out_path} --show")


if __name__ == "__main__":
    main()
