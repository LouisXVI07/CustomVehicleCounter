# Team Pupils — Classical Computer Vision Vehicle Counter
### Vehant Technologies 2-Day Hackathon | January 2026

This repository contains Team Pupils’ winning classical computer vision solution for accurate vehicle counting under strict constraints prohibiting deep learning. The system is fully interpretable, deterministic, and implemented entirely using classical image processing techniques without any trained models or neural networks.

---

## Problem Statement

The objective of the challenge was to:

- Count the total number of vehicles in a traffic video
- Use only classical image-processing techniques (OpenCV + NumPy)
- Process videos from a static camera with vehicles moving away from the lens
- Produce a single integer output
- Able to run fully offline without internet access
- Avoid all learning-based or pre-trained models

Explicitly prohibited:

- Deep learning detectors (YOLO, SSD, Faster R-CNN)
- Pretrained segmentation networks
- CNNs, embeddings, neural networks of any kind
- Hardcoding video-specific patterns

The approach required robust motion understanding, geometry analysis, and stable tracking without black-box inference.

---

## Solution Overview

The pipeline is divided into three primary stages:

1. Environmental Calibration
2. Real-Time Detection and Segmentation
3. Predictive Persistent Tracking and Count-Locking

The following sections describe each stage in detail.

---
![Workflow Diagram](assets/workflow.png)
---
## Stage 1 — Environment & Gate Calibration

### Temporal Median Background
A background model is constructed by sampling frames across the video, eliminating all moving vehicles. This creates a stable baseline for foreground extraction.

### Day/Night Auto-Detection
The system analyzes global brightness:
- Low brightness triggers Night Mode (CLAHE enhancement + heavier blurring)
- Normal brightness uses standard filtering

### Automatic Tripwire Calibration
Using Hough Line Transform on the background image:
- Detects road boundaries
- Computes road orientation
- Generates a dynamically angled virtual tripwire for counting

This ensures accurate counting regardless of camera tilt or lane direction.

---

## Stage 2 — Real-Time Detection Pipeline

### Preprocessing
The system applies:
- Grayscale conversion
- Gaussian blurring
- Frame differencing against the background model

### Adaptive Masking
Threshold calculations are based on environment mode (day or night), ensuring reliable blob extraction across lighting conditions.

### Elliptical Morphological Fusion
A large elliptical kernel is used to close gaps between fragmented vehicle regions. Unlike square kernels, the ellipse merges vertical splits (e.g., SUV bumper vs. rear window) while preserving separation between adjacent lanes.

### Multi-Tiered Geometric Profiling
Contours are evaluated using two profiles:

**Car Profile:**
- Large area
- Horizontal aspect ratio

**Bike Profile:**
- Smaller area
- Slightly vertical aspect ratio

This approach ensures sensitivity to motorcycles while avoiding pedestrian-like vertical shapes.

### Structural Sobel-Y Texture Verification
To avoid false positives:
- Sobel-Y operator measures horizontal edge intensity
- Vehicles exhibit strong, rigid horizontal gradients
- Shadows and pedestrians exhibit weak or vertical gradients

Only blobs that pass both geometric and texture filters qualify as valid vehicles.

---

## Stage 3 — Persistent Tracking & Counting

### Centroid Tracking
Each valid blob generates a centroid, which is linked to existing tracks through Euclidean matching.

### Velocity-Aided Prediction
Each track maintains:
- Estimated velocity vector (vx, vy)
- Predicted next position

This stabilizes tracking under partial occlusion or temporal mask flicker.

### ID Persistence
Tracks persist for up to 20 frames even when temporarily lost. This prevents:

- Mask fragmentation
- Double counting during flicker
- Loss due to momentary occlusion

### Tripwire Count Locking
When a tracked centroid crosses the dynamically aligned angled tripwire:
- The global count increments
- The ID is permanently locked
- It cannot be counted again under any conditions

This guarantees no double-counting even if the blob later splits or disappears.

---

## Performance Characteristics

The designed system demonstrates:

- Stable detection in day and night scenarios
- Accurate tracking under occlusion
- Correct handling of small motorcycles
- Prevention of SUV double-counting
- Full adherence to classical vision constraints
- No deep learning or video-specific tuning

The approach is transparent, reproducible, and robust across diverse traffic patterns.

---

## 📂 Repository Structure

```
CustomVehicleCounter/
├── main.py                # Main script containing Solution class
├── requirements.txt       # Python dependencies
├── assets/                # Images, workflow diagrams, demo videos
│   ├── workflow.png
│   ├── result_demo.mp4
│   └── video_output.mp4
└── README.md              # Project documentation
```
---
## 🔧 Installation

Install all required dependencies using:

```bash
pip install -r requirements.txt
```
---

## ▶️ Usage

Run the solution using the following code:

```python
from main import Solution

sol = Solution()

# The forward method returns the total integer count
count = sol.forward("path/to/traffic_video.avi")

print(f"Total Count: {count}")
```
---
