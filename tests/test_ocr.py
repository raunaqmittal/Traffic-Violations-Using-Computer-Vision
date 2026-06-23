"""Tests for OCR PlateReadResult and evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    ocr_accuracy,
    compute_iou,
    average_precision,
    classification_metrics,
)
from src.components.ocr.plate_reader import normalize_plate


class TestPlateNormalization:
    def test_valid_plate_passes(self):
        text, valid = normalize_plate("MH12AB1234")
        assert text == "MH12AB1234" and valid is True

    def test_strips_separators_and_case(self):
        text, valid = normalize_plate("ka-01 a 1234")
        assert text == "KA01A1234" and valid is True

    def test_short_series_plate(self):
        _, valid = normalize_plate("DL3CAB1234")
        assert valid is True

    def test_shop_sign_text_rejected(self):
        # Merged junk that is not a plate format must be flagged invalid.
        _, valid = normalize_plate("OPENSALE")
        assert valid is False

    def test_too_short_rejected(self):
        _, valid = normalize_plate("MH12")
        assert valid is False


class TestOCRAccuracy:
    def test_exact_match(self):
        assert ocr_accuracy(["MH12AB1234"], ["MH12AB1234"]) == 1.0

    def test_no_match(self):
        assert ocr_accuracy(["MH12AB1234"], ["KA01CD5678"]) == 0.0

    def test_partial_match_counts_correctly(self):
        gt = ["MH12AB1234", "KA01CD5678"]
        pr = ["MH12AB1234", "WRONG"]
        assert ocr_accuracy(gt, pr) == 0.5

    def test_case_insensitive(self):
        assert ocr_accuracy(["mh12ab1234"], ["MH12AB1234"]) == 1.0

    def test_empty_returns_zero(self):
        assert ocr_accuracy([], []) == 0.0


class TestIoU:
    def test_perfect_overlap(self):
        assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0, abs=1e-3)

    def test_no_overlap(self):
        assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0, abs=1e-3)

    def test_half_overlap(self):
        iou = compute_iou((0, 0, 10, 10), (5, 0, 15, 10))
        assert 0.3 < iou < 0.4


class TestAP:
    def test_perfect_detector(self):
        gt = [(0, 0, 10, 10), (20, 20, 30, 30)]
        pred = [(0, 0, 10, 10), (20, 20, 30, 30)]
        scores = [0.9, 0.8]
        ap = average_precision(pred, scores, gt, iou_threshold=0.5)
        assert ap == pytest.approx(1.0, abs=0.05)

    def test_no_predictions(self):
        gt = [(0, 0, 10, 10)]
        ap = average_precision([], [], gt)
        assert ap == 0.0
