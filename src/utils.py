from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2


VideoSource = Union[int, str]


def parse_source(value: str) -> VideoSource:
    """Turn '0' into webcam index 0; otherwise keep a path/URL string."""
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def open_video_source(source: VideoSource) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}. "
            "Check your webcam permissions or video path."
        )
    return cap


def ensure_output_path(path: str | None) -> Path | None:
    if not path:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
