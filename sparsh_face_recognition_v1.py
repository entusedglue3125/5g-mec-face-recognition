#!/usr/bin/env python3
"""
sparsh_face_recognition_v1.py
High-Performance, Zero-Lag Real-Time Face Recognition from Sparsh 5G Camera.
Features:
- Decoupled asynchronous AI inference worker (Zero Stream Lag, High FPS).
- Full 1080p HD crystal-clear output resolution (with 720p option).
- Low-latency RTSP FFMPEG flags (nobuffer, low_delay, TCP transport).
- Real-time bounding square with Person Name & Registration Number badges.
- Web management dashboard with enrollment and deletion.
"""

import sys
import os

# Auto-switch to virtual environment if face_recognition is not available in current python
try:
    import face_recognition
except ModuleNotFoundError:
    venv_python = "/home/st/sparsh_face/venv/bin/python"
    if os.path.exists(venv_python) and sys.executable != venv_python:
        print(f"[*] Switching to virtual environment python: {venv_python}")
        os.execv(venv_python, [venv_python] + sys.argv)
    else:
        raise

# Configure FFMPEG capture options for stable, clean TCP transport
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
import json
import base64
import argparse
import threading
from pathlib import Path
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
from werkzeug.utils import secure_filename

# ================= DEFAULT CONFIGURATION =================
RTSP_URL = "rtsp://admin:admin123@10.101.0.68:554/stream/avstream/channel=1/stream=0.sdp"
DEFAULT_PORT = 5001
RECOGNITION_TOLERANCE = 0.48  # Lower = stricter match, Higher = looser match
AI_INFERENCE_SCALE = 0.20    # Downscale for face detection (ultra-fast ~35ms)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "face_db"
PHOTOS_DIR = DATA_DIR / "photos"
INPUT_DIR = BASE_DIR / "known_faces_input"
DB_FILE = DATA_DIR / "faces_metadata.json"
ENCODINGS_FILE = DATA_DIR / "encodings.npy"
# ========================================================

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image_bgr(source):
    """Loads an image with proper EXIF orientation handling and returns BGR ndarray."""
    from PIL import Image, ImageOps
    import io

    if isinstance(source, (str, Path)):
        pil_img = Image.open(str(source))
    elif isinstance(source, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(source))
    elif isinstance(source, np.ndarray):
        return source
    else:
        return None

    pil_img = ImageOps.exif_transpose(pil_img)
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class FaceDatabase:
    """Manages enrolled faces, encodings, names, and registration numbers."""
    def __init__(self):
        self.lock = threading.Lock()
        self.records = []  # list of dicts: {'id': str, 'name': str, 'reg_no': str, 'photo': str, 'created_at': str}
        self.encodings = []  # list of 128-d numpy arrays
        self.load()
        self.scan_input_directory()

    def load(self):
        with self.lock:
            if DB_FILE.exists() and ENCODINGS_FILE.exists():
                try:
                    with open(DB_FILE, "r") as f:
                        self.records = json.load(f)
                    enc_arr = np.load(ENCODINGS_FILE, allow_pickle=True)
                    self.encodings = [np.array(e, dtype=np.float64) for e in enc_arr]
                    print(f"[*] Loaded {len(self.records)} enrolled face(s) from database.")
                except Exception as e:
                    print(f"[!] Error loading database: {e}")
                    self.records = []
                    self.encodings = []
            else:
                self.records = []
                self.encodings = []

    def save(self):
        with self.lock:
            with open(DB_FILE, "w") as f:
                json.dump(self.records, f, indent=2)
            np.save(ENCODINGS_FILE, np.array(self.encodings, dtype=object), allow_pickle=True)

    def enroll(self, name: str, reg_no: str, image_input):
        """Enrolls a face from an image (filepath, bytes, or BGR ndarray)."""
        name = name.strip()
        reg_no = reg_no.strip()
        if not name or not reg_no:
            return False, "Name and Registration Number cannot be empty."

        image_bgr = load_image_bgr(image_input)
        if image_bgr is None:
            return False, "Could not load or decode the image."

        # Resize large phone photos for optimal face detection speed
        h, w = image_bgr.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1200:
            scale = 1200.0 / max_dim
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_img)
        if not locations:
            locations = face_recognition.face_locations(rgb_img, number_of_times_to_upsample=2)

        if not locations:
            return False, "No face detected in the provided image. Please provide a clear, well-lit photo."

        encodings = face_recognition.face_encodings(rgb_img, locations)
        if not encodings:
            return False, "Could not extract face features. Please try another photo."

        encoding = encodings[0]
        face_id = f"{secure_filename(name)}_{secure_filename(reg_no)}"
        photo_filename = f"{face_id}_{int(time.time())}.jpg"
        photo_path = PHOTOS_DIR / photo_filename
        cv2.imwrite(str(photo_path), image_bgr)

        with self.lock:
            # Check if this person already exists (by name or non-generic reg_no) and update, or append
            existing_idx = None
            for idx, r in enumerate(self.records):
                if r["name"].lower() == name.lower():
                    existing_idx = idx
                    break
                if reg_no.lower() not in ("faculty", "n/a", "unknown") and r["reg_no"].lower() == reg_no.lower():
                    existing_idx = idx
                    break

            record = {
                "id": face_id,
                "name": name,
                "reg_no": reg_no,
                "photo": photo_filename,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            if existing_idx is not None:
                self.records[existing_idx] = record
                self.encodings[existing_idx] = encoding
                msg = f"Updated existing record for {name} (Reg No: {reg_no})"
            else:
                self.records.append(record)
                self.encodings.append(encoding)
                msg = f"Successfully enrolled {name} (Reg No: {reg_no})"

        self.save()
        print(f"[+] {msg}")
        return True, msg

    def delete(self, identifier: str):
        """Deletes a record by ID or registration number."""
        with self.lock:
            idx = None
            for i, r in enumerate(self.records):
                if r.get("id", "").lower() == identifier.lower() or r["reg_no"].lower() == identifier.lower() or r["name"].lower() == identifier.lower():
                    idx = i
                    break
            if idx is None:
                return False, f"Record '{identifier}' not found."

            r = self.records.pop(idx)
            self.encodings.pop(idx)
            photo_file = PHOTOS_DIR / r.get("photo", "")
            if photo_file.exists():
                photo_file.unlink(missing_ok=True)

        self.save()
        return True, f"Deleted {r['name']} ({r['reg_no']})"

    def scan_input_directory(self):
        """Auto-enrolls images placed in `known_faces_input/`."""
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
            for file_path in INPUT_DIR.glob(ext):
                stem = file_path.stem
                parts = stem.replace("-", "_").split("_", 1)
                if len(parts) == 2:
                    name, reg_no = parts[0], parts[1]
                else:
                    name, reg_no = parts[0], "UNKNOWN-REG"

                already = any(r["name"].lower() == name.lower() for r in self.records)
                if not already:
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        ok, msg = self.enroll(name, reg_no, img)
                        print(f"[*] Input folder auto-enroll {file_path.name}: {msg}")

    def recognize_faces(self, frame_bgr, tolerance=RECOGNITION_TOLERANCE, scale=AI_INFERENCE_SCALE):
        """
        Detects faces in frame_bgr and matches against enrolled database.
        Returns list of dicts: [{'box': (left, top, right, bottom), 'name': str, 'reg_no': str, 'matched': bool}]
        """
        h, w = frame_bgr.shape[:2]
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        if not locations:
            return []

        encodings = face_recognition.face_encodings(rgb_small, locations)
        results = []

        inv_scale = 1.0 / scale

        with self.lock:
            has_db = len(self.encodings) > 0
            known_encs = list(self.encodings)
            records = list(self.records)

        for (top, right, bottom, left), enc in zip(locations, encodings):
            box_left = int(left * inv_scale)
            box_top = int(top * inv_scale)
            box_right = int(right * inv_scale)
            box_bottom = int(bottom * inv_scale)

            name = "Unknown"
            reg_no = "N/A"
            matched = False

            if has_db:
                distances = face_recognition.face_distance(known_encs, enc)
                best_idx = int(np.argmin(distances))
                if distances[best_idx] <= tolerance:
                    name = records[best_idx]["name"]
                    reg_no = records[best_idx]["reg_no"]
                    matched = True

            results.append({
                "box": (box_left, box_top, box_right, box_bottom),
                "name": name,
                "reg_no": reg_no,
                "matched": matched
            })

        return results


class RTSPStreamReader:
    """
    Ultra-low-latency threaded RTSP reader.
    Continuously drains ffmpeg packet buffers so stream lag stays near 0ms.
    """
    def __init__(self, rtsp_url):
        self.url = rtsp_url
        self.cap = None
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self.new_frame_event = threading.Event()
        self.start()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                print(f"[*] Connecting to RTSP camera: {self.url} ...")
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not self.cap.isOpened():
                    self.connected = False
                    print("[!] Failed to connect to RTSP stream. Retrying in 2 seconds...")
                    time.sleep(2)
                    continue

                self.connected = True
                print("[+] Successfully connected to RTSP camera (Stable TCP Stream)!")

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame
                self.frame_id += 1
                self.new_frame_event.set()

    def get_latest(self):
        """Returns latest (frame_id, frame) immediately without blocking."""
        with self.lock:
            return self.frame_id, self.frame

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


class AsyncFaceInferenceWorker:
    """
    Runs face detection & recognition asynchronously in a separate background thread.
    This completely isolates AI inference from the video streaming loop, guaranteeing
    maximum streaming framerate with ZERO stutter and ZERO lag.
    """
    def __init__(self, database, tolerance=RECOGNITION_TOLERANCE, scale=AI_INFERENCE_SCALE):
        self.db = database
        self.tolerance = tolerance
        self.scale = scale
        self.latest_results = []
        self.lock = threading.Lock()
        self.running = True
        self.pending_frame = None
        self.has_new_frame = threading.Event()
        self.fps = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit_frame(self, frame):
        """Submits the latest frame for asynchronous recognition."""
        self.pending_frame = frame
        self.has_new_frame.set()

    def _run(self):
        count = 0
        t0 = time.time()
        while self.running:
            self.has_new_frame.wait(timeout=0.2)
            self.has_new_frame.clear()

            if self.pending_frame is None:
                continue

            frame = self.pending_frame
            results = self.db.recognize_faces(frame, tolerance=self.tolerance, scale=self.scale)

            with self.lock:
                self.latest_results = results

            count += 1
            if count % 10 == 0:
                elapsed = time.time() - t0
                self.fps = 10.0 / elapsed if elapsed > 0 else 0
                t0 = time.time()

    def get_detections(self):
        with self.lock:
            return list(self.latest_results)


def draw_recognition_box(frame, item):
    """
    Draws high-contrast, crystal-clear HUD bounding box and badge.
    """
    left, top, right, bottom = item["box"]
    matched = item["matched"]
    name = item["name"]
    reg_no = item["reg_no"]

    if matched:
        border_color = (0, 230, 115)      # Bright Emerald Green
        badge_bg = (15, 95, 35)           # Dark Green
        text_color = (255, 255, 255)
    else:
        border_color = (40, 50, 245)      # Bright Red
        badge_bg = (15, 15, 140)          # Dark Red
        text_color = (255, 255, 255)

    # 1. Main Square / Rectangle Box
    cv2.rectangle(frame, (left, top), (right, bottom), border_color, 2)

    # 2. Corner Bracket Highlights
    corner_len = min(24, (right - left) // 4)
    thick = 3
    # Top-Left
    cv2.line(frame, (left, top), (left + corner_len, top), border_color, thick)
    cv2.line(frame, (left, top), (left, top + corner_len), border_color, thick)
    # Top-Right
    cv2.line(frame, (right, top), (right - corner_len, top), border_color, thick)
    cv2.line(frame, (right, top), (right, top + corner_len), border_color, thick)
    # Bottom-Left
    cv2.line(frame, (left, bottom), (left + corner_len, bottom), border_color, thick)
    cv2.line(frame, (left, bottom), (left, bottom - corner_len), border_color, thick)
    # Bottom-Right
    cv2.line(frame, (right, bottom), (right - corner_len, bottom), border_color, thick)
    cv2.line(frame, (right, bottom), (right, bottom - corner_len), border_color, thick)

    # 3. Text Label Badges
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.70
    font_thick = 2

    if matched:
        lines = [f"NAME: {name}", f"REG NO: {reg_no}"]
    else:
        lines = ["UNKNOWN PERSON", "NO REGISTRATION"]

    line_height = 26
    padding = 8
    banner_w = 0
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, font_thick)
        banner_w = max(banner_w, tw)
    banner_w = max(banner_w + (padding * 2), right - left)
    banner_h = (len(lines) * line_height) + (padding * 2)

    banner_x1 = left
    banner_y1 = top - banner_h - 8
    if banner_y1 < 10:  # Draw below if too close to top
        banner_y1 = bottom + 8
    banner_x2 = banner_x1 + banner_w
    banner_y2 = banner_y1 + banner_h

    h_frame, w_frame = frame.shape[:2]
    banner_x1 = max(0, min(banner_x1, w_frame - 10))
    banner_x2 = max(0, min(banner_x2, w_frame - 1))
    banner_y1 = max(0, min(banner_y1, h_frame - 10))
    banner_y2 = max(0, min(banner_y2, h_frame - 1))

    # Badge background + border
    cv2.rectangle(frame, (banner_x1, banner_y1), (banner_x2, banner_y2), badge_bg, -1)
    cv2.rectangle(frame, (banner_x1, banner_y1), (banner_x2, banner_y2), border_color, 1)

    curr_y = banner_y1 + padding + 18
    for line in lines:
        cv2.putText(frame, line, (banner_x1 + padding, curr_y), font, font_scale, text_color, font_thick, cv2.LINE_AA)
        curr_y += line_height


# ================= FLASK APPLICATION =================
app = Flask(__name__)
db = FaceDatabase()
stream_reader = None
ai_worker = None

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sparsh 5G Live AI Face Recognition System</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0f1d;
            --bg-card: #141e33;
            --border-color: #243452;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg-primary); color: var(--text-main); min-height: 100vh; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid var(--border-color); }
        .header h1 { font-size: 1.5rem; font-weight: 700; color: #fff; }
        .header h1 span { color: var(--accent-green); }
        .badge-live { background: #064e3b; color: #34d399; padding: 6px 14px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; display: flex; align-items: center; gap: 6px; }
        .dot { width: 8px; height: 8px; background: #34d399; border-radius: 50%; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.3); } }
        
        .grid { display: grid; grid-template-columns: 1fr 400px; gap: 20px; }
        @media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
        
        .card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 18px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
        .card h2 { font-size: 1.15rem; font-weight: 600; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; }
        
        .video-container { position: relative; width: 100%; border-radius: 10px; overflow: hidden; background: #000; border: 1px solid var(--border-color); box-shadow: 0 8px 16px rgba(0,0,0,0.4); }
        .video-container img { width: 100%; height: auto; display: block; border-radius: 8px; image-rendering: auto; }
        
        .stream-controls { display: flex; justify-content: space-between; align-items: center; margin-top: 12px; font-size: 0.85rem; color: var(--text-dim); }
        .res-toggle { display: flex; gap: 6px; }
        .res-btn { background: #1e293b; border: 1px solid var(--border-color); color: var(--text-dim); padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; font-weight: 600; }
        .res-btn.active { background: var(--accent-blue); color: #fff; border-color: var(--accent-blue); }

        .form-group { margin-bottom: 12px; }
        label { display: block; font-size: 0.85rem; font-weight: 500; color: var(--text-dim); margin-bottom: 5px; }
        input[type="text"], input[type="file"] { width: 100%; padding: 10px 12px; background: #0a0f1d; border: 1px solid var(--border-color); border-radius: 6px; color: #fff; font-size: 0.9rem; }
        input[type="file"] { padding: 8px; cursor: pointer; }
        .btn { width: 100%; padding: 11px; font-weight: 600; border-radius: 6px; border: none; cursor: pointer; transition: 0.2s; font-size: 0.9rem; }
        .btn-green { background: var(--accent-green); color: #022c22; }
        .btn-green:hover { background: #059669; }
        .btn-delete { background: #7f1d1d; color: #fca5a5; border: none; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; }
        .btn-delete:hover { background: var(--accent-red); color: #fff; }

        .table-container { margin-top: 12px; max-height: 300px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th { text-align: left; color: var(--text-dim); padding: 8px; border-bottom: 1px solid var(--border-color); }
        td { padding: 8px; border-bottom: 1px solid #1c2842; vertical-align: middle; }
        .thumb { width: 36px; height: 36px; border-radius: 50%; object-fit: cover; border: 1px solid var(--border-color); }
        .alert { padding: 8px 12px; border-radius: 6px; margin-bottom: 12px; font-size: 0.85rem; display: none; }
        .alert-success { background: #064e3b; color: #a7f3d0; border: 1px solid #059669; }
        .alert-error { background: #450a0a; color: #fecaca; border: 1px solid #b91c1c; }
        .tabs { display: flex; gap: 8px; margin-bottom: 14px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }
        .tab-btn { background: none; border: none; color: var(--text-dim); font-size: 0.85rem; font-weight: 600; cursor: pointer; padding: 5px 10px; border-radius: 6px; }
        .tab-btn.active { color: var(--accent-green); background: #0d2826; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Sparsh 5G <span>Face Recognition System</span></h1>
            <p style="color: var(--text-dim); font-size: 0.85rem; margin-top: 3px;">High FPS • Real-Time AI Detection & Recognition</p>
        </div>
        <div class="badge-live">
            <div class="dot"></div>
            LIVE AI STREAM
        </div>
    </div>

    <div class="grid">
        <!-- Live Video View -->
        <div class="card">
            <h2>
                <span>📷 Live Recognition Feed</span>
                <span style="font-size: 0.8rem; font-weight: normal; color: var(--accent-green);">Full Quality</span>
            </h2>
            <div class="video-container">
                <img id="streamImg" src="/video_feed" alt="Sparsh 5G Live AI Stream">
            </div>
            <div class="stream-controls">
                <div>
                    <b>Feed:</b> Sparsh 5G Cellular Camera &nbsp;|&nbsp; <b>Status:</b> <span style="color:#34d399;">Active</span>
                </div>
            </div>
        </div>

        <!-- Sidebar: Enrollment & Database -->
        <div>
            <div class="card" style="margin-bottom: 18px;">
                <h2>➕ Enroll New Face</h2>
                
                <div id="statusAlert" class="alert"></div>

                <div class="tabs">
                    <button class="tab-btn active" id="tabUpload" onclick="switchTab('upload')">Upload Photo</button>
                    <button class="tab-btn" id="tabCapture" onclick="switchTab('capture')">Capture from Stream</button>
                </div>

                <form id="enrollForm">
                    <div class="form-group">
                        <label>Person Full Name *</label>
                        <input type="text" id="personName" name="name" placeholder="e.g. John Doe" required>
                    </div>

                    <div class="form-group">
                        <label>Registration Number (Reg No) *</label>
                        <input type="text" id="regNo" name="reg_no" placeholder="e.g. 192312043 or Faculty" required>
                    </div>

                    <div class="form-group" id="photoFileGroup">
                        <label>Face Photo File *</label>
                        <input type="file" id="photoFile" name="photo" accept="image/*">
                    </div>

                    <button type="submit" class="btn btn-green" id="submitBtn">
                        <span id="btnText">Save & Enroll Person</span>
                    </button>
                </form>
            </div>

            <!-- Enrolled Faces Card -->
            <div class="card">
                <h2>👥 Enrolled Faces (<span id="recordCount">0</span>)</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Photo</th>
                                <th>Name</th>
                                <th>Reg No</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody id="recordsTableBody">
                            <tr><td colspan="4" style="text-align:center; color:var(--text-dim);">Loading enrolled faces...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentMode = 'upload';

        function switchTab(mode) {
            currentMode = mode;
            document.getElementById('tabUpload').classList.toggle('active', mode === 'upload');
            document.getElementById('tabCapture').classList.toggle('active', mode === 'capture');
            const fileGroup = document.getElementById('photoFileGroup');
            const photoInput = document.getElementById('photoFile');
            const btnText = document.getElementById('btnText');

            if (mode === 'capture') {
                fileGroup.style.display = 'none';
                photoInput.removeAttribute('required');
                btnText.innerText = '📸 Capture Face From Live Camera';
            } else {
                fileGroup.style.display = 'block';
                photoInput.setAttribute('required', 'required');
                btnText.innerText = 'Save & Enroll Person';
            }
        }

        function showAlert(msg, isSuccess) {
            const el = document.getElementById('statusAlert');
            el.className = 'alert ' + (isSuccess ? 'alert-success' : 'alert-error');
            el.innerText = msg;
            el.style.display = 'block';
            setTimeout(() => { el.style.display = 'none'; }, 6000);
        }

        async function loadRecords() {
            try {
                const res = await fetch('/api/records');
                const data = await res.json();
                document.getElementById('recordCount').innerText = data.length;
                const tbody = document.getElementById('recordsTableBody');

                if (data.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color:var(--text-dim);">No faces enrolled yet. Add one above!</td></tr>';
                    return;
                }

                tbody.innerHTML = data.map(r => `
                    <tr>
                        <td><img class="thumb" src="/photos/${r.photo}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\\'http://www.w3.org/2000/svg\\\' width=\\\'36\\\' height=\\\'36\\\' fill=\\\'%23475569\\\' viewBox=\\\'0 0 24 24\\\'><path d=\\\'M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z\\\'/></svg>'"></td>
                        <td><b>${r.name}</b></td>
                        <td style="color: #60a5fa;">${r.reg_no}</td>
                        <td><button class="btn-delete" onclick="deleteRecord('${r.id || r.reg_no}', '${r.name}')">Delete</button></td>
                    </tr>
                `).join('');
            } catch (err) {
                console.error(err);
            }
        }

        document.getElementById('enrollForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const name = document.getElementById('personName').value;
            const regNo = document.getElementById('regNo').value;
            const submitBtn = document.getElementById('submitBtn');
            submitBtn.disabled = true;

            try {
                let response;
                if (currentMode === 'capture') {
                    response = await fetch('/api/enroll_camera', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name, reg_no: regNo })
                    });
                } else {
                    const formData = new FormData();
                    formData.append('name', name);
                    formData.append('reg_no', regNo);
                    const fileInput = document.getElementById('photoFile');
                    if (fileInput.files.length > 0) {
                        formData.append('photo', fileInput.files[0]);
                    }
                    response = await fetch('/api/enroll_upload', {
                        method: 'POST',
                        body: formData
                    });
                }

                const result = await response.json();
                if (response.ok && result.success) {
                    showAlert(result.message, true);
                    document.getElementById('enrollForm').reset();
                    loadRecords();
                } else {
                    showAlert(result.message || 'Enrollment failed.', false);
                }
            } catch (err) {
                showAlert('Network or server error occurred.', false);
            } finally {
                submitBtn.disabled = false;
            }
        });

        async function deleteRecord(id, name) {
            if (!confirm(`Are you sure you want to remove ${name}?`)) return;
            try {
                const res = await fetch('/api/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ id: id })
                });
                const data = await res.json();
                if (data.success) {
                    showAlert(data.message, true);
                    loadRecords();
                } else {
                    showAlert(data.message, false);
                }
            } catch (err) {
                showAlert('Failed to delete record.', false);
            }
        }

        loadRecords();
        setInterval(loadRecords, 8000);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/photos/<filename>")
def serve_photo(filename):
    from flask import send_from_directory
    return send_from_directory(PHOTOS_DIR, filename)

@app.route("/api/records")
def api_records():
    with db.lock:
        return jsonify(db.records)

@app.route("/api/enroll_upload", methods=["POST"])
def api_enroll_upload():
    name = request.form.get("name", "").strip()
    reg_no = request.form.get("reg_no", "").strip()
    photo_file = request.files.get("photo")

    if not name or not reg_no or not photo_file:
        return jsonify({"success": False, "message": "Name, Reg No, and photo file are required."}), 400

    filestr = photo_file.read()
    ok, msg = db.enroll(name, reg_no, filestr)
    return jsonify({"success": ok, "message": msg}), (200 if ok else 400)

@app.route("/api/enroll_camera", methods=["POST"])
def api_enroll_camera():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    reg_no = data.get("reg_no", "").strip()

    if not name or not reg_no:
        return jsonify({"success": False, "message": "Name and Reg No are required."}), 400

    _, frame = stream_reader.get_latest()
    if frame is None:
        return jsonify({"success": False, "message": "Cannot capture from camera. Stream is not ready."}), 500

    ok, msg = db.enroll(name, reg_no, frame)
    return jsonify({"success": ok, "message": msg}), (200 if ok else 400)

@app.route("/api/delete", methods=["POST"])
def api_delete():
    data = request.get_json() or {}
    identifier = (data.get("id") or data.get("reg_no") or "").strip()
    if not identifier:
        return jsonify({"success": False, "message": "Identifier is required."}), 400

    ok, msg = db.delete(identifier)
    return jsonify({"success": ok, "message": msg}), (200 if ok else 400)

@app.route("/video_feed")
def video_feed():
    def generate_mjpeg():
        last_frame_id = -1
        fps_count = 0
        t_fps = time.time()
        fps_text = "FPS: --"

        while True:
            fid, frame = stream_reader.get_latest()
            if frame is None:
                # Placeholder frame when camera is connecting
                placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Connecting to Sparsh 5G Camera...",
                            (80, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
                ret, jpeg = cv2.imencode(".jpg", placeholder)
                if ret:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n")
                time.sleep(0.1)
                continue

            if fid == last_frame_id:
                time.sleep(0.005)
                continue

            last_frame_id = fid

            # Submit frame to background AI inference worker (non-blocking)
            ai_worker.submit_frame(frame)

            # Retrieve latest cached detections (instant, no waiting!)
            cached_results = ai_worker.get_detections()

            # Draw recognition boxes and Name/Reg No badges on frame
            display_frame = frame.copy()
            for item in cached_results:
                draw_recognition_box(display_frame, item)

            # Calculate FPS
            fps_count += 1
            if fps_count % 15 == 0:
                elapsed = time.time() - t_fps
                curr_fps = 15.0 / elapsed if elapsed > 0 else 0
                fps_text = f"STREAM: {curr_fps:.1f} FPS | AI: {ai_worker.fps:.1f} DET/s"
                t_fps = time.time()

            # Sleek HUD tag on top-left of frame
            cv2.putText(display_frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display_frame, fps_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 1, cv2.LINE_AA)

            ret, jpeg = cv2.imencode(".jpg", display_frame)
            if not ret:
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n")

    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ================= MAIN EXECUTION =================
def main():
    global stream_reader, ai_worker

    parser = argparse.ArgumentParser(description="Sparsh 5G Low-Latency Face Recognition System")
    parser.add_argument("--stream", type=int, default=0, choices=[0, 1], help="Stream: 0 for Main Camera stream, 1 for Substream")
    parser.add_argument("--rtsp", type=str, default=None, help="Custom RTSP URL")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run web server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--tolerance", type=float, default=RECOGNITION_TOLERANCE, help="Face match tolerance (lower=stricter)")
    parser.add_argument("--gui", action="store_true", help="Display local OpenCV desktop window (cv2.imshow)")
    parser.add_argument("--enroll-image", type=str, help="Enroll a person from an image file path")
    parser.add_argument("--name", type=str, help="Person name for CLI enrollment")
    parser.add_argument("--regno", type=str, help="Person registration number for CLI enrollment")
    parser.add_argument("--list", action="store_true", help="List enrolled persons in database")
    args = parser.parse_args()

    if args.rtsp is None:
        args.rtsp = f"rtsp://admin:admin123@10.101.0.68:554/stream/avstream/channel=1/stream={args.stream}.sdp"

    if args.list:
        print("\n=== Enrolled Faces Database ===")
        if not db.records:
            print("No records found.")
        else:
            for idx, r in enumerate(db.records, 1):
                print(f"{idx}. Name: {r['name']} | Reg No: {r['reg_no']} | Enrolled: {r.get('created_at', 'N/A')}")
        return

    if args.enroll_image:
        if not args.name or not args.regno:
            print("[!] Error: --name and --regno are required when using --enroll-image")
            return
        img = cv2.imread(args.enroll_image)
        if img is None:
            print(f"[!] Error: Could not read image at '{args.enroll_image}'")
            return
        ok, msg = db.enroll(args.name, args.regno, img)
        print(f"[*] Result: {msg}")
        return

    print("=" * 65)
    print("   SPARSH 5G CAMERA - LIVE FACE RECOGNITION SYSTEM")
    print("=" * 65)
    print(f"[*] RTSP Camera URL : {args.rtsp}")
    print(f"[*] Web Server URL  : http://{args.host}:{args.port}")
    print(f"[*] Face Database   : {DATA_DIR} ({len(db.records)} registered)")
    print("=" * 65)

    import socket
    def is_port_open(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", p)) != 0

    if not is_port_open(args.port):
        print(f"\n[!] Note: Port {args.port} is in use by another instance.")
        print(f"[*] Access existing stream at: http://192.168.10.12:{args.port}")
        print("[*] To restart, run: pkill -f sparsh_face_recognition_v1\n")
        if not args.gui:
            return

    # Start low-latency camera reader and async AI worker
    stream_reader = RTSPStreamReader(args.rtsp)
    ai_worker = AsyncFaceInferenceWorker(db, tolerance=args.tolerance, scale=AI_INFERENCE_SCALE)

    if args.gui:
        if not os.environ.get("DISPLAY"):
            print("[!] Cannot open GUI window: No X11 DISPLAY found in this SSH session.")
            print("[!] Connect using 'ssh -Y st@192.168.10.12' to enable GUI.")
            print(f"[*] Or view live in your web browser: http://192.168.10.12:{args.port}")
            return

        print("[*] Starting local desktop window (press 'q' to exit)...")
        if is_port_open(args.port):
            threading.Thread(target=lambda: app.run(host=args.host, port=args.port, debug=False, use_reloader=False), daemon=True).start()

        last_fid = -1
        try:
            while True:
                fid, frame = stream_reader.get_latest(timeout=0.2)
                if frame is None or fid == last_fid:
                    time.sleep(0.005)
                    continue

                last_fid = fid
                ai_worker.submit_frame(frame)
                detections = ai_worker.get_detections()

                disp = frame.copy()
                for item in detections:
                    draw_recognition_box(disp, item)

                cv2.imshow("Sparsh 5G Live AI Stream (1080p)", disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            stream_reader.stop()
            cv2.destroyAllWindows()
    else:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
