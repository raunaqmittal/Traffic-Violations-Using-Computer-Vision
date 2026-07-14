"""
Evaluate the seatbelt classifier (seatbelt_finetuned.pt) on the val split.

Usage:
    python evaluate_seatbelt.py

Outputs per-class Precision, Recall, and overall Top-1 Accuracy against
data/seatbelt datasets/final_dataset/val/.
"""
import glob
import math
import os

from ultralytics import YOLO


def evaluate(
    model_path: str = r"models\weights\seatbelt_finetuned.pt",
    val_dir: str = r"data\seatbelt datasets\final_dataset\val",
    positive_class: str = "no_seatbelt",
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation dataset not found: {val_dir}")

    model = YOLO(model_path)

    tp = fp = fn = tn = correct = total = 0
    classes = os.listdir(val_dir)
    print(f"Classes found: {classes}")

    for cls in classes:
        img_paths = glob.glob(os.path.join(val_dir, cls, "*.*"))
        for img_path in img_paths:
            results = model(img_path, verbose=False)[0]
            pred_cls = results.names[results.probs.top1]
            total += 1
            if pred_cls == cls:
                correct += 1
            if cls == positive_class:
                if pred_cls == positive_class:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred_cls == positive_class:
                    fp += 1
                else:
                    tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    top1_acc  = correct / total if total > 0 else 0.0

    print(f"\n{'='*54}")
    print(f"  Positive class : '{positive_class}'")
    print(f"  True Positives : {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Negatives : {tn}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  Top-1 Accuracy : {top1_acc:.4f}  ({correct}/{total})")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    evaluate()
