Team Pupils — Classical CV Vehicle Analytics
Vehant Technologies 2-Day Hackathon | January 2026

This repository contains Team Pupils' winning classical computer vision solution for the Vehant Technologies Vehicle Count Challenge.
The core objective was to build a vehicle counting system using only classical image processing—no deep learning, no pre-trained models, no neural networks.

Our pipeline blends geometry, texture analysis, predictive tracking, and adaptive morphology to achieve a highly stable counting system even under real-world distortions.

📌 Problem Statement & Requirements

The challenge was to estimate the total vehicle count in a traffic video where:

Vehicles move away from a static camera

Only classical CV techniques were allowed

The solution must run offline using standard libraries

Output must be a single integer: the total vehicle count

Restrictions included:

❌ No YOLO / SSD / Faster R-CNN / Deep Learning

❌ No pre-trained segmentation models

❌ No learned weights of any kind

❌ No hardcoding for specific videos

🛠️ Unified Processing Pipeline

Our solution uses a temporal–spatial CV pipeline, combining background modeling, geometric reasoning, Sobel texture verification, and predictive tracking.

The complete logic flow is visualized below (use your own Mermaid diagram file or embed GitHub-rendered block):

graph TD
    classDef init fill:#f8f9fa,stroke:#343a40,stroke-width:2px,color:#212529;
    classDef loop fill:#e7f5ff,stroke:#228be6,stroke-width:2px,color:#212529;
    classDef logic fill:#f4fce3,stroke:#5c940d,stroke-width:2px,color:#212529;
    classDef final fill:#fff5f5,stroke:#fa5252,stroke-width:3px,color:#212529;

    subgraph Initialization ["Step 1: Environmental Calibration"]
        A[Input Video File] --> B[Temporal Median Background Modeling]
        B --> C{Detect Environment}
        C -->|Avg Brightness < 65| D[Night Mode: CLAHE + High Blur]
        C -->|Avg Brightness >= 65| E[Day Mode: Standard Filters]
        B --> F[Hough Line Transform: Road Edge Detection]
        F --> G[Calibrate Dynamic Angled Tripwire]
    end

    subgraph Loop ["Step 2: Real-Time Detection & Tracking"]
        H[Read Next Frame] --> I[Gaussian Blur & Frame Differencing]
        I --> J[Adaptive Thresholding & Morphological Fusion]
        J --> K[Contour Extraction]
        
        K --> L{Multi-Tiered Classification}
        L -->|Area/Aspect Match| M[Structural Texture Verification: Sobel-Y]
        L -->|Reject Noise| H
        
        M -->|Valid Vehicle| N[Centroid Generation]
        M -->|Invalid/Human| H
        
        N --> O[Velocity-Aided Temporal Tracking]
        O --> P{Cross Angled Tripwire?}
        P -->|Yes & Not Counted| Q[Increment Count & Lock ID]
        P -->|No / Already Counted| R[Update Movement Vector]
    end

    Q --> S{Stream End?}
    R --> S
    S -->|More Frames| H
    S -->|Video End| T[Return Total Integer Count]

    class A,B,C,D,E,F,G init;
    class H,I,J,K,O loop;
    class L,M,P logic;
    class T final;

⚖️ Constraints & Challenges Faced
1. Deep Learning Prohibited

We had to engineer a fully classical CV solution capable of:

Detecting vehicles of varying classes

Avoiding false positives like pedestrians

Handling light changes and fragmentation

2. Vehicle Fragmentation

SUVs often produce multiple disconnected blobs in classical differencing.

3. Small Motorcycle Detection

Tiny bikes resemble noise in frame differencing.

4. Perspective Distortion

Vehicles shrink and slow visually as they move away from the camera, requiring dynamic size scaling.

💡 Key Innovations by Team Pupils
1. Structural Sobel-Y Texture Verification

Problem: Shadows & pedestrians trigger false detections
Solution: Apply Sobel-Y operator to detect horizontal metal-like structure
Why it works: Vehicles have strong rigid horizontal gradients; pedestrians do not.

2. Multi-Tiered Geometric Profiling

Problem: Single area threshold fails for motorcycles
Solution: Dual profiles

Cars → high area, wide aspect

Bikes → smaller area, vertical aspect

Benefit: High sensitivity to bikes while rejecting human-sized noise.

3. Adaptive Elliptical “Super-Fusion”

Problem: Mask fragmentation in SUVs
Solution: Elliptical morphological closing (15×15 to 23×23)
Why: Ellipses merge vertical splits but avoid merging across lanes.

4. ID-Locked Predictive Tracking

Problem: Flicker or occlusion causes double-counting
Solution:

Velocity prediction (vx, vy)

Sticky persistent IDs

Boolean count-lock flag

Result: No duplicate counts—even with mask flicker.

📈 Performance Demonstration
System Architecture (static diagram placeholder)

Add your workflow diagram here:

![Workflow](workflow.png)

Real-Time Tracking Results

Our dynamically angled tripwire automatically aligns with the detected road geometry and works across lanes.

Demo Video 1 – Daylight / Standard Flow
<video src="./demo_day.mp4" width="480" controls></video>

Demo Video 2 – High Density Traffic
<video src="./demo_day_2.mp4" width="480" controls></video>


Replace video paths with your actual filenames when uploading.

🛠️ Setup & Execution
1. Install Requirements
pip install -r requirements.txt

2. Run the solution
python main.py --video path/to/demo_video.avi

3. Python Usage
from main import Solution

sol = Solution()
count = sol.forward("path/to/traffic_video.mp4")
print("Total Count:", count)

📂 Repository Structure
├── main.py                  # Contains Solution class and pipeline
├── utils/                   # Helper modules (tracking, geometry, filtering)
├── workflow.png             # Workflow diagram (add your file)
├── demo_day.mp4             # Video demo (replace with real file)
├── demo_day_2.mp4           # Second demo video
├── requirements.txt
└── README.md

👥 Team Information

Team Name: Pupils
Hackathon: Vehant 2-Day CV Challenge (Jan 2026)
