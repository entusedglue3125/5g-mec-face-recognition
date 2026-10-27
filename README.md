# 5G-MEC Edge Face Recognition & Surveillance Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![5G-MEC](https://img.shields.io/badge/Network-5G%20%2F%20MEC%20Edge-FF6F00)](https://www.etsi.org/technologies/multi-access-edge-computing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A low-latency, real-time edge face recognition and surveillance analytics pipeline designed for deployment across 5G Multi-Access Edge Computing (MEC) servers, supporting high-throughput RTSP video streams from IP and drone cameras.

---

## 🌟 Overview

This project was built for edge surveillance and access control in 5G/6G experimental networks. Operating on edge compute nodes connected to Sparsh and CDAC network surveillance streams, it achieves zero-lag face recognition, automated metadata matching, and web-based management.

### Key Capabilities
- **Decoupled Asynchronous Inference**: Employs background worker threads with double-buffered frame queues to eliminate RTSP network buffer backlog and maintain maximum streaming FPS.
- **Hardware-Accelerated RTSP Pipeline**: Custom FFMPEG capture parameters (`rtsp_transport: tcp`, `nobuffer`, `low_delay`) eliminating packet tear and visual artifacts.
- **Biometric Embeddings Database**: 128-dimensional deep metric face embeddings stored in `face_db/encodings.npy` alongside student/personnel registration metadata (`faces_metadata.json`).
- **Web-Based Management Interface**: Embedded HTTP dashboard providing live camera feeds, visual bounding box overlays, and real-time face enrollment/deletion.
- **Full HD Resolution**: Supports 1080p crystal-clear video inference with adaptive downscaling for bandwidth-constrained links.

---

## 📁 Repository Structure

```text
5g-mec-face-recognition/
├── sparsh_face_recognition_v1.py   # Primary RTSP capture, worker threads & HTTP server
├── face_db/
│   ├── encodings.npy               # Pre-computed 128D facial vector embeddings
│   └── faces_metadata.json         # Personnel metadata (name, registration ID, timestamps)
├── requirements.txt                # Python package dependencies
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start

### 1. Prerequisites
Install system libraries for `dlib` and OpenCV:
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev
```

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/entusedglue3125/5g-mec-face-recognition.git
cd 5g-mec-face-recognition

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launching the Surveillance Engine
Run with default RTSP or camera parameters:
```bash
python3 sparsh_face_recognition_v1.py --rtsp "rtsp://username:password@192.168.10.12:554/stream1" --port 8080
```
Open **[http://localhost:8080](http://localhost:8080)** to access the live edge surveillance monitor and personnel enrollment panel.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
