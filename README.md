<<<<<<< HEAD
## Real-Time Accident Detection

Real-time road accident detection system using a custom YOLOv8 model for crash detection, combined with vehicle counting and traffic density estimation on video streams.

## Project Overview

This repository contains:
- A custom-trained YOLOv8 model to detect `crash` events.
- Inference scripts for image and video accident detection.
- A video pipeline that overlays:
  - accident status (`ACCIDENT!` / `NO ACCIDENT`)
  - vehicle count
  - estimated traffic density (`Low`, `Medium`, `High`)

The crash model was trained on the Roboflow dataset:
- Project: `car-crash-0eqcb`
- Classes: `1` (`crash`)
- Link: [Roboflow Dataset](https://universe.roboflow.com/thnh-snbl5/car-crash-0eqcb/dataset/1)

## Features

- Crash detection with custom YOLOv8 weights (`best.pt`)
- Vehicle detection using pretrained `yolov8n.pt`
- Live frame-by-frame annotation with OpenCV
- Traffic density estimation based on detected vehicle count
- Output video generation with overlays for monitoring and analysis

## Repository Structure

```text
.
|-- README.md
|-- YOLO_main.py                 # Train custom crash detector
|-- YOLOv8n.py                   # Predict on test images
|-- Video_pipeline.py            # Full video inference pipeline
|-- YOLO/
|   `-- Car Crash.v1i.yolov8/
|       `-- data.yaml            # Dataset config
|-- runs/
|   `-- detect/train-2/
|       |-- weights/best.pt      # Trained crash model
|       `-- results.csv          # Training metrics
|-- Test_img_YOLO/               # Test images
|-- Videos/                      # Input videos
`-- Outputs/                     # Output videos
```

## Model Training

Training is done with Ultralytics YOLO:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="D:\\Real Time Accident Detection\\YOLO\\Car Crash.v1i.yolov8\\data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
```

Main training script: `YOLO_main.py`

## Inference

### 1) Image Inference

Script: `YOLOv8n.py`

- Loads trained weights from `runs/detect/train-2/weights/best.pt`
- Runs prediction on images from `Test_img_YOLO`
- Saves prediction outputs

### 2) Video Inference Pipeline

Script: `Video_pipeline.py`

Pipeline:
- Detect crashes with custom model (`best.pt`)
- Detect vehicles with `yolov8n.pt`
- Count vehicle classes: bicycle, car, motorcycle, bus, truck
- Estimate traffic density:
  - `< 5` vehicles: `Low`
  - `< 15` vehicles: `Medium`
  - `>= 15` vehicles: `High`
- Save annotated output video to `Outputs/output_video.mp4`

## Training Performance (train-2)

From `runs/detect/train-2/results.csv` (epoch 50):
- Precision: `0.882`
- Recall: `0.842`
- mAP@50: `0.855`
- mAP@50-95: `0.500`

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/<your-username>/real-time-accident-detection.git
cd real-time-accident-detection
```

### 2) Create and activate virtual environment (recommended)

```bash
python -m venv .venv
```

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install ultralytics opencv-python
```

If you want notebook/statistical analysis support:

```bash
pip install notebook pandas matplotlib seaborn
```

## Usage

Run image detection:

```bash
python YOLOv8n.py
```

Run full video pipeline:

```bash
python Video_pipeline.py
```

Train/retrain model:

```bash
python YOLO_main.py
```

## Notes for Publishing to GitHub

- Replace hardcoded local paths in scripts with relative paths before sharing publicly.
- Add a `requirements.txt` for reproducible setup.
- Do not commit large model files unless needed; use Releases or Git LFS for heavy artifacts.
- Add sample output images/videos in the README for better project presentation.

## Future Improvements

- Real-time webcam or CCTV stream support
- Alert system (email/SMS/dashboard trigger) on accident detection
- Tracking-based vehicle counting for improved density estimation
- Deployment as a lightweight API or web dashboard

## Author

Created by **anindyapaul* for accident detection and traffic monitoring research.
paulanindya2001@gmail.com

=======
# smart-traffic-accident-detector
>>>>>>> e6afa60e2535e7beac09e94e794d8afc96e412e0
