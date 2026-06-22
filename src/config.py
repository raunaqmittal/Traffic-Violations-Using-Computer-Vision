"""
Configuration loader.
Reads all YAML config files once and exposes typed access.
"""

import yaml
from pathlib import Path
from typing import Any


_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _load(filename: str) -> dict[str, Any]:
    path = _CONFIG_DIR / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_pipeline() -> dict[str, Any]:
    return _load("pipeline.yaml")


def load_cameras() -> list[dict[str, Any]]:
    data = _load("cameras.yaml")
    return data.get("cameras", [])


def load_violations() -> dict[str, Any]:
    data = _load("violations.yaml")
    return data.get("violations", {})


def load_tracker_config() -> dict[str, Any]:
    data = _load("violations.yaml")
    return data.get("tracker", {})


def get_camera_config(camera_id: str) -> dict[str, Any] | None:
    for cam in load_cameras():
        if cam["id"] == camera_id:
            return cam
    return None
