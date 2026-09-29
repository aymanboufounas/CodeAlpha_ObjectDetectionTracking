# CodeAlpha Object Detection & Tracking

Real-time **object detection and multi-object tracking** project created for **CodeAlpha Artificial Intelligence Internship — Task 4**.

The application takes live webcam input or a video file, detects objects using a pretrained **YOLOv8** model, and tracks them across frames using **Deep SORT**. Each tracked object receives a persistent tracking ID while it remains visible.

## CodeAlpha Task 4 Requirements

| Requirement | Implementation |
|---|---|
| Real-time video input | OpenCV webcam or video file |
| Pretrained object detector | YOLOv8 |
| Detect objects frame by frame | Ultralytics YOLO |
| Draw bounding boxes and labels | OpenCV |
| Object tracking | Deep SORT |
| Unique tracking IDs | Displayed above each tracked object |
| Real-time visualization | OpenCV window |

## Features

- Live webcam detection
- Video-file detection
- YOLOv8 object detection
- Deep SORT multi-object tracking
- Persistent tracking IDs
- Class labels
- Detection confidence
- FPS counter
- Number of tracked objects
- Optional result video export
- Configurable confidence threshold
- Clean modular Python structure

## Project Structure

```text
CodeAlpha_ObjectDetectionTracking/
├── main.py
├── requirements.txt
├── README.md
├── PROJECT_INFO.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── detector.py
│   ├── tracker.py
│   └── utils.py
├── tests/
│   └── test_utils.py
└── samples/
    └── README.md
```

## How the AI Pipeline Works

```text
Camera / Video
      │
      ▼
 OpenCV Frame
      │
      ▼
 YOLOv8 Detection
      │
      ├── Bounding box
      ├── Class
      └── Confidence
      │
      ▼
 Deep SORT Tracker
      │
      ├── Motion tracking
      ├── Appearance features
      └── Track association
      │
      ▼
 Persistent Object ID
      │
      ▼
 Annotated Live Video
```

## Installation

Python **3.10 or 3.11** is recommended.

### 1. Clone the repository

```bash
git clone https://github.com/aymanboufounas/CodeAlpha_ObjectDetectionTracking.git
cd CodeAlpha_ObjectDetectionTracking
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

The first run downloads the small pretrained `yolov8n.pt` model automatically.

## Run With Webcam

```bash
python main.py
```

or explicitly:

```bash
python main.py --source 0
```

Press **Q** or **ESC** to stop.

## Run With a Video

```bash
python main.py --source samples/demo.mp4
```

## Save the Processed Video

```bash
python main.py --source samples/demo.mp4 --output outputs/result.mp4
```

## Change Detection Confidence

```bash
python main.py --source 0 --conf 0.55
```

## Use Another YOLOv8 Model

For better accuracy at the cost of speed:

```bash
python main.py --model yolov8s.pt
```

Typical choices:

- `yolov8n.pt` — fastest and smallest
- `yolov8s.pt` — better accuracy, still lightweight
- `yolov8m.pt` — more accurate but heavier

## Example Output

A detected person can appear as:

```text
person | ID 3 | 0.91
```

A car can appear as:

```text
car | ID 8 | 0.87
```

The **ID** is produced by Deep SORT and is used to follow the same object across successive frames.

## Technologies

- Python
- OpenCV
- Ultralytics YOLOv8
- Deep SORT
- deep-sort-realtime
- Computer Vision
- Multi-Object Tracking

## Testing

```bash
pytest
```

## Notes

This project uses a pretrained YOLO model rather than training an object detector from scratch, which matches the CodeAlpha task requirement to use a pretrained model such as YOLO or Faster R-CNN.

The YOLO model file is excluded from Git because it can be downloaded automatically.

## Author

**Ayman Boufounas**

Built for the CodeAlpha Artificial Intelligence Internship.
