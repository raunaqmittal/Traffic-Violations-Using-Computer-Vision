"""
Signal state detection using HSV thresholding on a fixed ROI.
Returns: "red" | "green" | "unknown"

Used by both stop_line.py and red_light.py.
"""

import cv2
import numpy as np


def detect_signal_state(
    frame: np.ndarray,
    signal_roi: tuple[int, int, int, int],
    red_pixel_fraction: float = 0.15,
    green_pixel_fraction: float = 0.10,
) -> str:
    x1, y1, x2, y2 = signal_roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_pixels = crop.shape[0] * crop.shape[1]

    # Red wraps around the HSV hue circle
    red_mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
    red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)

    green_mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([90, 255, 255]))
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels / total_pixels >= red_pixel_fraction:
        return "red"
    if green_pixels / total_pixels >= green_pixel_fraction:
        return "green"
    return "unknown"
