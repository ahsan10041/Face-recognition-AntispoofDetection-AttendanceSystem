# Face Recognition + Anti-Spoofing Attendance System

A real-time biometric attendance system built with Python and Flask. It combines face detection, face recognition, and passive presentation attack detection (PAD) to reject photo and screen-replay attacks without requiring any user cooperation.

## Demo

![Live Demo](demo/demo.png)

## Features

- Real-time face detection via MediaPipe BlazeFace
- Face recognition using DeepFace / FaceNet embeddings (cosine similarity)
- Passive anti-spoofing via DeepFace FasNet (MiniFASNet V2 + V1SE ensemble) — single-frame, no user cooperation required
- Live video feed annotated with green (Real) / red (Spoof) bounding boxes
- Web interface for user registration, attendance marking, and log viewing
- Attendance log persisted to JSON, including anti-spoof decision metadata per entry

## Anti-Spoofing Approach

The anti-spoofing module (`src/anti_spoof.py`) uses DeepFace's bundled FasNet implementation, accessed via `DeepFace.extract_faces(anti_spoofing=True)`.

FasNet is an ensemble of two MiniFASNet variants ([Liu et al., 2021](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)):

| Model | Crop scale | What it captures |
|---|---|---|
| MiniFASNetV2 | 2.7× | Screen pixel grid, emission patterns, brightness uniformity |
| MiniFASNetV1SE | 4.0× | Flat-object geometry, large-area lighting distribution |

Each model outputs a two-class softmax; the ensemble produces a single `antispoof_score ∈ [0, 1]`:

```
antispoof_score > 0.5  →  Real
antispoof_score ≤ 0.5  →  Spoof
```

Empirically observed score separation on a Logitech C920 webcam:

| Presentation type | Typical score | Decision |
|---|---|---|
| Live face | 0.95 – 0.99 | Real ✓ |
| Phone screen / printed photo | 0.28 – 0.35 | Spoof ✓ |

The ~0.60 margin around the 0.5 threshold gives robust discrimination under typical office lighting with no overlap between genuine and attack score distributions.

A background worker thread runs FasNet every 400 ms, decoupled from the MJPEG stream so the video feed is never blocked by inference. Anti-spoofing is skipped when multiple faces are in frame (a yellow box is shown instead) to prevent score misattribution between subjects.

## Installation

Requires Python 3.10 or 3.11.

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# PyTorch — CPU-only build is sufficient for FasNet and avoids a large CUDA download
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The MediaPipe BlazeFace model (`blaze_face_short_range.tflite`) and FasNet weights (~10 MB each) are downloaded automatically on first run.

## Running

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

> **Note:** The first startup takes 30–40 s while PyTorch loads the FasNet model weights. Subsequent runs are faster once the OS has cached the files. A warm-up pass runs automatically before the server begins accepting requests.

## Usage

1. **Register** — enter a name and click "Register User". Stand in front of the camera alone.
2. **Mark Attendance** — click "Scan Face". The system detects, anti-spoof-checks, and recognises you.
3. **Attendance Log** — visible in the bottom panel; also saved to `data/attendance_log.json`. Each entry includes the anti-spoof label, confidence, and raw `antispoof_score`.
4. **Quit** — click the "Quit" button in the header to gracefully stop the server (useful on Windows where Ctrl+C may not reach the terminal).

The live feed shows a green bounding box (Real) or red (Spoof) at all times. Multi-face frames show a yellow box and skip PAD.

## Project Structure

```
├── app.py                      Flask application and REST endpoints
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── anti_spoof.py           DeepFace FasNet wrapper (production anti-spoofing)
│   ├── anti_spoof_classical.py Self-implemented classical CV pipeline (Phase 2, retained for reference)
│   ├── face_detector.py        MediaPipe BlazeFace wrapper
│   └── face_recognizer.py      DeepFace / FaceNet wrapper
├── models/
│   └── blaze_face_short_range.tflite   Downloaded automatically on first run
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/main.js
├── data/                       Created at runtime (user DB + attendance log)
└── demo/
    └── demo.png
```

## Configuration

The face recognition similarity threshold can be adjusted in `src/face_recognizer.py`:

```python
def __init__(self, db_path="data/users.pkl", threshold=0.6):
```

The EER occurs at τ ≈ 0.52. The default τ = 0.6 biases towards lower FAR (fewer impostors accepted) at the cost of slightly higher FRR — appropriate for an attendance context where falsely recording an absent person as present is the more serious error.

The anti-spoofing threshold (0.5) is internal to DeepFace's FasNet implementation and is not user-configurable. The empirically observed score separation (~0.60 margin) means adjusting it is not necessary in practice.

## Troubleshooting

**Real face flagged as spoof**
- Ensure even, adequate lighting — extreme shadows or direct backlighting confuse FasNet
- Make sure only one face is in frame (multi-face frames show a yellow box and skip PAD entirely)
- Move slightly closer to the camera so the face fills more of the frame

**Photo / screen not being rejected**
- Ensure anti-spoofing is enabled (it is on by default)
- FasNet's 0.5 threshold is not adjustable; if a high-quality attack passes, the limitation is the model's training coverage rather than the threshold setting

**Camera not detected**
- The app tries index 1 (external webcam) before index 0 (built-in). Ensure your webcam is connected before starting.

**Slow first startup (30–40 s)**
- PyTorch is loading the FasNet model weights. Subsequent startups are faster once the OS has cached the files.

## Dependencies

| Package | Purpose |
|---|---|
| flask | Web server, REST API, MJPEG routing |
| opencv-python | Video capture, image processing |
| mediapipe | Face detection (BlazeFace) |
| deepface | Face recognition (FaceNet) + anti-spoofing (FasNet) |
| torch | PyTorch backend for FasNet inference |
| tf-keras | TensorFlow/Keras backend for DeepFace FaceNet |
| numpy | Array operations |
| scikit-learn | Cosine similarity |
| scikit-image | LBP feature extraction (classical CV reference module) |
| Pillow | Image I/O |
