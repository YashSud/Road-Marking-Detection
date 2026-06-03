# Road Marking Detection

A real-time road marking detection desktop application powered by **YOLOv8** and a custom-trained model. The app features a **Tkinter GUI** that allows users to run detection on both images and videos, with color-coded bounding boxes and live detection readouts for 12 road marking classes.

## Overview

This project uses a fine-tuned YOLOv8 model (`best.pt`) to detect and classify road markings from user-provided images or video files. Each class is assigned a distinct color for visual clarity. Detected results are displayed in a pop-up window, listed in the GUI, and saved automatically as output files.

## Project Structure

```
Road-Marking-Detection-main/
├── my project.py    # Main application script
└── best.pt          # Custom-trained YOLOv8 weights
```

## Detectable Classes

| Class | Color |
|---|---|
| Dashed And Solid Yellow Lines | Purple |
| Dashed White Lines | Blue-Orange |
| Dashed Yellow Lines | Orange |
| Do Not Enter | Deep Pink |
| Double Solid White Lines | Silver |
| Go Straight | Spring Green |
| Go Straight or Left Turn | Cornflower Blue |
| Go Straight or Right Turn | Forest Green |
| Left Turn | Crimson |
| No Parking | Dark Red |
| Stop | Orange-Red |
| Zebra Crossing | Steel Blue |

## Features

- **Tkinter GUI** — simple desktop interface with file browser
- **Image Detection** — detects all road markings in a static image and saves `<filename>_detected.jpg`
- **Video Detection** — processes video frame-by-frame and saves `<filename>_detected.mp4`
- **Live Detection List** — GUI panel updates in real time with detected class names and confidence scores
- **Color-coded Bounding Boxes** — each class rendered in a unique color for easy identification
- **ESC to Exit** — press Escape during video playback to stop processing

## Requirements

- Python 3.x
- ultralytics (YOLOv8)
- opencv-python
- Pillow
- tkinter (built into Python standard library)

Install dependencies with:

```bash
pip install ultralytics opencv-python pillow
```

## Usage

1. Update the model path in `my project.py` to point to `best.pt` on your machine:

```python
model = YOLO(r"path\to\best.pt")
```

2. Run the application:

```bash
python "my project.py"
```

3. Click **"Detect from Image/Video"** in the GUI, select your file, and view the results.

## Outputs

- **Image:** annotated image saved as `<original_name>_detected.jpg` in the same directory
- **Video:** annotated video saved as `<original_name>_detected.mp4` in the same directory
- **GUI Panel:** live list of detected class names and confidence scores

## Key Concepts

| Concept | Purpose |
|---|---|
| YOLOv8 | Real-time object detection model by Ultralytics |
| Custom Weights (`best.pt`) | Fine-tuned on road marking dataset for domain-specific detection |
| Confidence Score | Probability assigned by the model to each detected bounding box |
| Tkinter | Python's built-in GUI toolkit used to build the desktop interface |
| OpenCV | Handles image/video I/O, bounding box drawing, and result saving |
