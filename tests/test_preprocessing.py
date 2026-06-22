"""Tests for frame preprocessing module."""

import numpy as np
import pytest
from src.components.preprocessing.frame_processor import process_frame, _laplacian_variance


def _make_frame(brightness: int = 120, noise: bool = False) -> np.ndarray:
    frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
    if noise:
        frame = frame + np.random.randint(0, 20, frame.shape, dtype=np.uint8)
    return frame


def test_process_frame_returns_same_shape():
    frame = _make_frame()
    result, quality = process_frame(frame)
    assert result.shape == frame.shape


def test_dark_frame_increases_brightness():
    dark = _make_frame(brightness=30)
    processed, _ = process_frame(dark)
    assert processed.mean() > dark.mean()


def test_blurry_frame_detected():
    blurry = np.full((480, 640, 3), 128, dtype=np.uint8)   # uniform = zero gradient = very blurry
    _, quality = process_frame(blurry, blur_threshold=100.0)
    assert quality.is_blurry is True


def test_sharp_frame_not_flagged():
    sharp = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, quality = process_frame(sharp, blur_threshold=10.0)
    assert quality.is_blurry is False


def test_rain_filter_applied_flag():
    frame = _make_frame()
    _, quality = process_frame(frame, apply_rain_filter=True)
    assert quality.rain_filter_applied is True


def test_no_rain_filter_by_default():
    frame = _make_frame()
    _, quality = process_frame(frame)
    assert quality.rain_filter_applied is False
