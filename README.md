# CyberTriageTool

# A Digital Forensics Application for Video Analysis
# Table of Contents
1. Introduction
2. Features
3. System Requirements
4. Installation
5. Usage
6. Output Details
7. Troubleshooting

# Introduction
The NUCES Video Triage Tool is a forensic application designed for investigators and analysts to process video files efficiently. It allows users to extract metadata, detect faces, and generate snapshots from videos to facilitate rapid evidence review in digital investigations.

This tool is particularly useful for handling large video files, enabling quick and automated processing to focus on critical elements like face detection.

# Features
Metadata Extraction: Retrieves information such as resolution, duration, frame rate, and file creation dates.
Face Detection: Automatically detects and saves individual faces from video frames using advanced algorithms.
Snapshot Generation: Extracts snapshots at defined intervals for quick visual review.
Video Playback: Built-in support for video playback to examine specific segments.
User-Friendly Interface: Simple and intuitive UI for seamless navigation.

# System Requirements
To run the NUCES Video Triage Tool, ensure the following:

Hardware Requirements
Processor: Intel Core i5 or equivalent
RAM: 4 GB or more
Disk Space: 1 GB free

Software Requirements
Operating System: Windows 10 or later
Python Version: 3.10 or later

# Installation
Step 1: Install Python
Download Python from the official site: https://www.python.org/.
Run the installer and check the option Add Python to PATH during installation.

Step 2: Install Dependencies
Open the Command Prompt and run the following commands:
pip install tkinter
pip install opencv-python
pip install opencv-python-headless
pip install pillow
pip install face-detection

Step 3: Download the Tool

# Usage
Step 1: Run the Application
1. Navigate to the folder containing main_code.py.
2. Open the tool by running the following command in the Command Prompt
  python main_code.py
3. The application window will appear, ready for use.

Step 2: Analyze a Video
Upload a video file (supported formats: .mp4, .avi, .mov) using the interface.
The tool will automatically process the video to:
1. Extract metadata
2. Detect faces
3. Generate snapshots

Step 3: View Results
1. Metadata Report: Click the View Metadata button to see details of the uploaded video.
2. Face Detection Results: Click the View Detected Faces button to browse through images of detected faces.
3. Snapshots: Browse snapshots in the video_analysis_output folder.

# Output Details
1. Detected Faces: Saved in the video_analysis_output/faces directory as .jpg files.
2. Snapshots: Stored in video_analysis_output/snapshots.
3. Metadata: Displayed in-app and saved in video_analysis_output/metadata.txt.

# Troubleshooting
1. Python Not Found: Ensure Python is installed and added to your PATH.
2. Missing Dependencies: Run the following command to reinstall required libraries
  pip install -r requirements.txt
3. Face Detection Issues: Verify that the face-detection library is installed.

   


