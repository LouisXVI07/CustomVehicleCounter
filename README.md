Team Pupils: Classical CV Vehicle AnalyticsVehant Technologies 2-Day Hackathon | Jan 2026📌 Problem Statement & ObjectivesThe goal of this challenge is to estimate the total count of vehicles in a traffic video feed using strictly classical image processing1. All vehicles are moving away from a fixed-position camera2. The solution must be self-contained and operate without any reliance on deep learning, neural networks, or pre-trained models333333333.🛠️ Unified Processing PipelineOur solution, developed by Team Pupils, utilizes a sophisticated temporal-spatial pipeline to solve the counting problem without "black box" models.Code snippetgraph TD
    %% Global Styles
    classDef init fill:#f8f9fa,stroke:#343a40,stroke-width:2px,color:#212529;
    classDef loop fill:#e7f5ff,stroke:#228be6,stroke-width:2px,color:#212529;
    classDef logic fill:#f4fce3,stroke:#5c940d,stroke-width:2px,color:#212529;
    classDef final fill:#fff5f5,stroke:#fa5252,stroke-width:3px,color:#212529;

    %% Initialization Phase
    subgraph Initialization ["Step 1: Environmental Calibration"]
        A[Input Video File] --> B[Temporal Median Background Modeling]
        B --> C{Detect Environment}
        C -->|Avg Brightness < 65| D[Night Mode: CLAHE + High Blur]
        C -->|Avg Brightness >= 65| E[Day Mode: Standard Filters]
        B --> F[Hough Line Transform: Road Edge Detection]
        F --> G[Calibrate Dynamic Angled Tripwire]
    end

    %% Main Processing Loop
    subgraph Loop ["Step 2: Real-Time Detection & Tracking"]
        H[Read Next Frame] --> I[Gaussian Blur & Frame Differencing]
        I --> J[Adaptive Thresholding & Morphological Fusion]
        J --> K[Contour Extraction]
        
        %% Geometric Classification
        K --> L{Multi-Tiered Classification}
        L -->|Area/Aspect Match| M[Structural Texture Verification: Sobel-Y]
        L -->|Reject Noise| H
        
        M -->|Valid Vehicle| N[Centroid Generation]
        M -->|Invalid/Human| H
        
        %% Persistent Tracking
        N --> O[Velocity-Aided Temporal Tracking]
        O --> P{Cross Angled Tripwire?}
        P -->|Yes & Not Counted| Q[Increment Count & Lock ID]
        P -->|No / Already Counted| R[Update Movement Vector]
    end

    %% Final Tally
    Q --> S{Stream End?}
    R --> S
    S -->|More Frames| H
    S -->|Video End| T[Return Total Integer Count]

    %% Applying Classes
    class A,B,C,D,E,F,G init;
    class H,I,J,K,O loop;
    class L,M,P logic;
    class T final;
⚖️ Constraints & Challenges FacedDeep Learning Prohibition: The strict ban on learning-based models required us to build a robust classifier from scratch using geometry and texture4444.Vehicle Fragmentation: Higher-contrast vehicles (like SUVs) often create "broken" masks in classical differencing where the roof and bumper appear separate.Motorcycle Sensitivity: Small bikes are often filtered out as noise, or confused with pedestrians.Perspective Distortion: Objects appear smaller and move slower as they move further away, requiring dynamic threshold scaling.💡 Key Innovations by Team Pupils1. Structural Sobel-Y Texture VerificationTo strictly exclude pedestrians and environmental noise, we implemented a structural texture check.The Logic: Vehicles possess rigid, metallic horizontal edges (bumpers, trunks, spoilers).The Solution: We apply a Sobel-Y operator to every candidate blob. If the horizontal gradient intensity does not meet a specific threshold, the object is discarded as non-metallic (e.g., a pedestrian or shadow).2. Multi-Tiered Geometric ProfilingTo solve the "missing bike" problem, we moved beyond a single area threshold.The Solution: We established distinct profiles for Cars (high area, horizontal aspect) and Bikes (lower area, vertical-leaning aspect). This allows for high bike sensitivity while the verticality limit strictly rejects human-shaped contours.3. Adaptive Elliptical "Super-Fusion"The Innovation: We use a large, oriented Elliptical Morphological Kernel for closing operations.The Logic: This bridges the gap between fragmented vehicle parts (like the rear window vs. spare tire on an SUV) into a single solid blob, preventing a single car from being assigned multiple tracking points.4. ID-Locked Persistence TrackingThe Solution: Our tracker uses Velocity Prediction ($vx, vy$) and a Boolean Count Flag.The Logic: Once a unique vehicle ID crosses the tripwire and is tallied, it is permanently locked. Even if the mask flickers or splits later, the system cannot trigger a double count for that specific ID.📈 Performance DemonstrationSystem ArchitectureThe following static diagram outlines the high-level logic flow used during development:Real-Time Tracking ResultsOur Dynamic Angled Gate automatically aligns the tripwire with the road's specific perspective to ensure lane-independent accuracy.Video Demo 1: Primary Lane FlowVideo Demo 2: High-Density Tracking<video src="./demo_day.mp4" width="400" controls></video><video src="./demo_day_2.mp4" width="400" controls></video>🛠️ Setup & Execution1. RequirementsInstall the necessary classical processing libraries:Bashpip install -r requirements.txt
2. ImplementationThe solution follows the mandatory structure provided by the organizers5555.Pythonfrom main import Solution

sol = Solution()
# The forward method returns the total integer count [cite: 82]
count = sol.forward("path/to/traffic_video.avi") 
print(f"Total Count: {count}")
Team Name: Pupils
