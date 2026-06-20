"""
Performance evaluation utilities.
Computes: Precision, Recall, F1 (per violation type),
          mAP@0.5 (for detectors), OCR exact-match accuracy, FPS.
"""

import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }


def ocr_accuracy(ground_truth: list[str], predictions: list[str]) -> float:
    if not ground_truth:
        return 0.0
    correct = sum(1 for gt, pr in zip(ground_truth, predictions) if gt.strip().upper() == pr.strip().upper())
    return correct / len(ground_truth)


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def average_precision(
    pred_boxes: list[tuple],
    pred_scores: list[float],
    gt_boxes: list[tuple],
    iou_threshold: float = 0.5,
) -> float:
    if not gt_boxes:
        return 0.0
    if not pred_boxes:
        # No detections -> zero recall -> AP is 0 (guard against the
        # interpolation endpoints producing a phantom 0.5 area).
        return 0.0

    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = set()

    for i, pb in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[i] = 1
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / (len(gt_boxes) + 1e-6)

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[1], precision, [0]])
    for k in range(len(precision) - 2, -1, -1):
        precision[k] = max(precision[k], precision[k + 1])
    # np.trapz was removed in NumPy 2.0; np.trapezoid replaces it.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(precision, recall))


class FPSTimer:
    def __init__(self):
        self._start = None
        self._count = 0

    def start(self):
        self._start = time.perf_counter()
        self._count = 0

    def tick(self):
        self._count += 1

    def fps(self) -> float:
        if self._start is None or self._count == 0:
            return 0.0
        elapsed = time.perf_counter() - self._start
        return self._count / elapsed if elapsed > 0 else 0.0
