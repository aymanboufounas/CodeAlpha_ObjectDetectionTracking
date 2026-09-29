from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .detector import YOLODetector
from .tracker import ObjectTracker
from .utils import ensure_output_path, open_video_source, parse_source


def draw_track(frame, tracked):
    x1, y1, x2, y2 = tracked.xyxy

    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

    label = f"{tracked.class_name} | ID {tracked.track_id}"
    if tracked.confidence > 0:
        label += f" | {tracked.confidence:.2f}"

    (tw, th), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2,
    )

    label_top = max(0, y1 - th - baseline - 8)
    cv2.rectangle(
        frame,
        (x1, label_top),
        (min(width - 1, x1 + tw + 10), y1),
        (0, 220, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 5, max(th + 2, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def build_writer(cap, output_path: Path):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def run(
    source: str,
    model: str,
    confidence: float,
    output: str | None,
    show_fps: bool,
):
    video_source = parse_source(source)

    detector = YOLODetector(
        model_path=model,
        confidence=confidence,
    )
    tracker = ObjectTracker()

    cap = open_video_source(video_source)
    output_path = ensure_output_path(output)
    writer = build_writer(cap, output_path) if output_path else None

    previous_time = time.perf_counter()

    print("Object Detection + Deep SORT Tracking")
    print("Press Q or ESC to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.detect(frame)
            tracked_objects = tracker.update(frame, detections)

            for tracked in tracked_objects:
                draw_track(frame, tracked)

            current_time = time.perf_counter()
            elapsed = max(current_time - previous_time, 1e-6)
            fps = 1.0 / elapsed
            previous_time = current_time

            if show_fps:
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"Tracked objects: {len(tracked_objects)}",
                (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            cv2.imshow(
                "CodeAlpha - Object Detection & Tracking",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time YOLOv8 object detection with Deep SORT tracking."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0, 1, ...) or path to a video file.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model file. yolov8n.pt is downloaded automatically on first run.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Detection confidence threshold (default: 0.40).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output video path, e.g. outputs/result.mp4.",
    )
    parser.add_argument(
        "--hide-fps",
        action="store_true",
        help="Do not display FPS.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 < args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1.")

    run(
        source=args.source,
        model=args.model,
        confidence=args.conf,
        output=args.output,
        show_fps=not args.hide_fps,
    )


if __name__ == "__main__":
    main()
