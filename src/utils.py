from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2

VideoSource = Union[int, str]


def parse_source(value: str) -> VideoSource:
    """Convert nonnegative camera indices to integers, preserving video paths."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("Video source must not be empty.")
    if stripped.isdecimal():
        return int(stripped)
    return stripped


def open_video_source(source: VideoSource) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open video source: {source}. "
            "Check your webcam permissions or video path."
        )
    return cap


def ensure_output_path(path: str | None) -> Path | None:
    if not path:
        return None
    output = Path(path).expanduser()
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output video path must end in .mp4.")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
