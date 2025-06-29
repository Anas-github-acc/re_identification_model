# Cross-Camera Player Mapping and Tracking

This project provides a comprehensive solution for detecting, tracking, and associating individual players across two different camera views (e.g., a tactical view and a broadcast view) of a sports event. It leverages the power of Ultralytics YOLO for player detection and tracking, OpenCV for homography transformation, and the Hungarian algorithm for robust player identity matching.

## Overview

The core idea is to establish a geometric relationship (Homography) between the two camera perspectives using static field points. Once this relationship is known, player positions from one camera can be projected onto the other. Player identities are then matched across cameras frame-by-frame using a combination of spatial proximity (after projection) and visual appearance (color histograms). The final output is a combined video showing both views with consistent player IDs highlighted.

## Features

- Player Detection & Tracking: Utilizes a fine-tuned Ultralytics YOLO model with ByteTrack/BotSort for robust player identification and persistent tracking IDs within each video stream.
- Data Caching: Automatically saves and loads processed player tracks (.pkl files) to avoid redundant video processing on subsequent runs.
- Interactive Homography Point Selection: A separate tool (homography_selector.py) allows users to visually select corresponding static points across different frames of the tactical and broadcast videos, enabling accurate homography calculation even if common points aren't visible in the first frame.
- Homography Calculation: Computes a 3x3 homography matrix using OpenCV's findHomography with RANSAC for robust estimation.
- Cross-Camera Player Matching: Implements a cost-based matching strategy using the Hungarian algorithm (Linear Sum Assignment) considering:
- Spatial Cost: Distance between a player's projected position from one camera and the detected position in the other.
- Visual Cost: Similarity of player jersey color histograms.
- Combined Video Output: Generates a side-by-side video displaying both tactical and broadcast views, with bounding boxes, unique player IDs, and lines connecting matched players across cameras.

## Setup before running the model

## Installation

1. **Clone the repository and set up other necesary files as per need:**


   files you need to add:

   - **model -** Add the model best.pt `model/best.pt` (a fine-tuned version of Ultralytics YOLOv11)
   - **data -** add the videos to the data folder and do not forget to re-check the name convention  
   

   ```
   .
   ├── main_cross_camera_analysis.py  # Renamed main script for clarity
   ├── homography_selector.py
   ├── src/
   │   ├── data/
   │   │   ├── broadcast.mp4
   |   |   ├── broadcast_tracks_v2.pkl (Generated cached tracks)
   |   |   ├── tacticam_tracks_v2.pkl (Generated cached tracks)
   │   │   └── tacticam.mp4
   │   └── model/
   │       └── best.pt                  # Your fine-tuned Ultralytics YOLO model
   └── combined_output.mp4             # (Generated output video)

   ```
2. **Install Dependencies:**
   Open your terminal or command prompt and run:

   - Create the virtual environment and activate it
   ```bash
   uv venv .venv
   source .venv/bash/activate # Linux
   .venv/Scripts/activate # Windows
   ```
   
   - Install the dependencies
   ```bash
   uv sync
   ```

   - Run the script
   ```bash
   pytohn src/main.py
   ```

## License

-- NIL --
