"""
Image preprocessing pipeline.
Runs on every frame before detection.
Returns the processed frame along with quality metadata.
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class FrameQuality:
    is_blurry: bool
    blur_score: float
    rain_filter_applied: bool


def process_frame(
    frame: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    blur_threshold: float = 100.0,
    apply_rain_filter: bool = False,
) -> tuple[np.ndarray, FrameQuality]:
    """
    Apply preprocessing to a single BGR frame.

    Steps:
      1. CLAHE on luminance channel for low-light / shadow correction.
      2. Laplacian variance blur detection.
      3. Optional rain mitigation (median blur + unsharp mask).

    Returns the processed frame and a FrameQuality struct.
    """
    # Measure blur on the ORIGINAL frame. Running it after CLAHE inflates local
    # contrast and the Laplacian variance, so genuinely blurry frames would
    # falsely pass the sharpness threshold.
    blur_score = _laplacian_variance(frame)
    is_blurry = blur_score < blur_threshold

    processed = _apply_clahe(frame, clahe_clip_limit, clahe_tile_size)

    rain_applied = False
    if apply_rain_filter:
        processed = _rain_filter(processed)
        rain_applied = True

    quality = FrameQuality(
        is_blurry=is_blurry,
        blur_score=blur_score,
        rain_filter_applied=rain_applied,
    )
    return processed, quality


def _apply_clahe(frame: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _laplacian_variance(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _rain_filter(frame: np.ndarray) -> np.ndarray:
    blurred = cv2.medianBlur(frame, 3)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(blurred, -1, kernel)
