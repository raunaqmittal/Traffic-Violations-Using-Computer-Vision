"""
Interactive zone and line drawing tool for camera setup.
Run this once per camera to define:
  - Stop line
  - No-parking zone polygons
  - Signal ROI

Usage: python scripts/draw_zones.py --video data/samples/test_video.mp4

Controls:
  L = draw line (stop line)
  P = draw polygon (no-parking zone)
  R = draw rectangle (signal ROI)
  Z = undo last point
  Enter = finalize current shape
  Q = quit and print config
"""

import cv2
import argparse
import numpy as np
import json


current_points: list = []
shapes: list = []
mode: str = "idle"
frame_display: np.ndarray | None = None


def mouse_callback(event, x, y, flags, param):
    global current_points, frame_display
    if event == cv2.EVENT_LBUTTONDOWN and mode != "idle":
        current_points.append([x, y])


def draw_overlay(frame: np.ndarray) -> np.ndarray:
    img = frame.copy()
    for shape in shapes:
        pts = shape["points"]
        color = {"line": (0, 255, 0), "polygon": (255, 100, 0), "roi": (0, 200, 255)}.get(shape["type"], (255, 255, 255))
        if shape["type"] == "line" and len(pts) >= 2:
            cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, 2)
        elif shape["type"] == "polygon":
            poly = np.array(pts, np.int32)
            cv2.polylines(img, [poly], True, color, 2)
        elif shape["type"] == "roi" and len(pts) >= 2:
            cv2.rectangle(img, tuple(pts[0]), tuple(pts[1]), color, 2)

    for pt in current_points:
        cv2.circle(img, tuple(pt), 5, (0, 0, 255), -1)
    if len(current_points) > 1:
        for i in range(len(current_points) - 1):
            cv2.line(img, tuple(current_points[i]), tuple(current_points[i + 1]), (0, 0, 255), 1)
    return img


def print_config(shapes: list):
    stop_line = None
    signal_roi = None
    no_parking_zones = []
    for i, s in enumerate(shapes):
        if s["type"] == "line":
            stop_line = s["points"]
        elif s["type"] == "roi":
            p = s["points"]
            signal_roi = [p[0][0], p[0][1], p[1][0], p[1][1]]
        elif s["type"] == "polygon":
            no_parking_zones.append({"name": f"Zone_{i}", "polygon": s["points"]})
    print("\n# Paste into src/configs/cameras.yaml:\n")
    if stop_line:
        print(f"    stop_line: {stop_line}")
    if signal_roi:
        print(f"    signal_roi: {signal_roi}")
    if no_parking_zones:
        print("    no_parking_zones:")
        for z in no_parking_zones:
            print(f"      - name: \"{z['name']}\"")
            print(f"        polygon: {z['polygon']}")


def main():
    global mode, current_points, frame_display
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read video.")
        return

    frame_display = frame.copy()
    cv2.namedWindow("Zone Setup")
    cv2.setMouseCallback("Zone Setup", mouse_callback)

    print("Controls: L=stop line | P=no-parking polygon | R=signal ROI | Enter=finalize | Z=undo | Q=quit+print")

    while True:
        display = draw_overlay(frame_display)
        status = f"Mode: {mode} | Points: {len(current_points)} | Shapes: {len(shapes)}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Zone Setup", display)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("l"):
            mode = "line"
            current_points = []
        elif key == ord("p"):
            mode = "polygon"
            current_points = []
        elif key == ord("r"):
            mode = "roi"
            current_points = []
        elif key == ord("z") and current_points:
            current_points.pop()
        elif key == 13:  # Enter
            if current_points:
                shapes.append({"type": mode, "points": list(current_points)})
                current_points = []
                mode = "idle"
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    print_config(shapes)


if __name__ == "__main__":
    main()
