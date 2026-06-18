from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import threading
import time
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.anti_spoof import AntiSpoof
import json
from datetime import datetime
import os

app = Flask(__name__)

face_detector   = FaceDetector()
face_recognizer = FaceRecognizer()
anti_spoof      = AntiSpoof()

camera             = None
current_mode       = 'attendance'
attendance_log     = []
anti_spoof_enabled = True
last_debug_scores  = {}

# Shared state for background anti-spoof worker
_latest_frame  = None
_latest_faces  = []
_frame_lock    = threading.Lock()
_spoof_cache   = None
_spoof_lock    = threading.Lock()


def _spoof_worker():
    """
    Background thread: runs FasNet on single-face frames only.
    Multi-face frames are skipped — we can't reliably attribute a score
    to each face when two people are in frame simultaneously.
    """
    global _spoof_cache
    while True:
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None
            faces = list(_latest_faces)

        # Only predict when exactly one face is visible
        if frame is not None and len(faces) == 1 and anti_spoof_enabled:
            result = anti_spoof.predict(frame, faces[0])
            with _spoof_lock:
                _spoof_cache = result

        time.sleep(0.4)


# Thread is started in __main__ after model warm-up (see bottom of file)


def get_camera():
    global camera
    if camera is None:
        for index in [1, 0]:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                camera = cap
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Using camera index {index}")
                break
    return camera


def generate_frames():
    global _latest_frame, _latest_faces
    while True:
        cam = get_camera()
        success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        faces = face_detector.detect(frame)

        # Feed background worker
        with _frame_lock:
            _latest_frame = frame.copy()
            _latest_faces = list(faces)

        with _spoof_lock:
            cached = _spoof_cache

        for bbox in faces:
            x, y, w, h = bbox
            if len(faces) == 1 and cached and anti_spoof_enabled:
                frame = anti_spoof.annotate_frame(frame, bbox, cached)
            else:
                # Multiple faces or no result yet — neutral box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    global last_debug_scores
    cam = get_camera()
    success, frame = cam.read()
    if not success:
        return jsonify({'success': False, 'message': 'Failed to capture frame'})

    frame = cv2.flip(frame, 1)
    faces = face_detector.detect(frame)

    if len(faces) == 0:
        return jsonify({'success': False, 'message': 'No face detected'})
    if len(faces) > 1:
        return jsonify({'success': False, 'message': 'Multiple faces detected'})

    face_img = face_detector.extract_face(frame, faces[0])
    if face_img is None:
        return jsonify({'success': False, 'message': 'Failed to extract face'})

    if anti_spoof_enabled:
        spoof = anti_spoof.predict(frame, faces[0])
        last_debug_scores = spoof
        if not spoof['is_real']:
            return jsonify({
                'success': False,
                'message': f"Spoof detected ({spoof['confidence']:.0%})",
                'spoof_detected': True,
            })

    ret, buffer = cv2.imencode('.jpg', face_img)
    return jsonify({'success': True, 'face_data': buffer.tobytes().hex(),
                    'bbox': faces[0]})


@app.route('/register', methods=['POST'])
def register():
    data      = request.json
    name      = data.get('name', '').strip()
    face_data = data.get('face_data')

    if not name:
        return jsonify({'success': False, 'message': 'Name is required'})
    if not face_data:
        return jsonify({'success': False, 'message': 'Face data is required'})

    face_bytes = bytes.fromhex(face_data)
    face_array = np.frombuffer(face_bytes, dtype=np.uint8)
    face_img   = cv2.imdecode(face_array, cv2.IMREAD_COLOR)
    success, message = face_recognizer.register_user(name, face_img)
    return jsonify({'success': success, 'message': message})


@app.route('/recognize', methods=['POST'])
def recognize():
    global last_debug_scores
    cam = get_camera()
    success, frame = cam.read()
    if not success:
        return jsonify({'success': False, 'message': 'Failed to capture frame'})

    frame = cv2.flip(frame, 1)
    faces = face_detector.detect(frame)

    if len(faces) == 0:
        return jsonify({'success': False, 'message': 'No face detected'})
    if len(faces) > 1:
        return jsonify({'success': False, 'message': 'Multiple faces detected'})

    face_img = face_detector.extract_face(frame, faces[0])
    if face_img is None:
        return jsonify({'success': False, 'message': 'Failed to extract face'})

    if anti_spoof_enabled:
        # Fresh inference — model is warm so this takes ~200-400 ms, not 40 s
        spoof = anti_spoof.predict(frame, faces[0])
        last_debug_scores = spoof
        if not spoof['is_real']:
            return jsonify({
                'success':         False,
                'message':         f"Spoof detected ({spoof['confidence']:.0%})",
                'spoof_detected':  True,
                'antispoof_score': spoof['scores'].get('antispoof_score', 0),
            })

    name, confidence = face_recognizer.recognize(face_img)
    if name is None:
        return jsonify({'success': False, 'message': 'Unknown user',
                        'confidence': float(confidence)})

    timestamp = datetime.now()
    log_entry = {
        'name':       name,
        'timestamp':  timestamp.isoformat(),
        'confidence': float(confidence),
        'mode':       current_mode,
    }
    if anti_spoof_enabled:
        log_entry['anti_spoof'] = {
            'label':      spoof['label'],
            'confidence': spoof['confidence'],
            'score':      spoof['scores'].get('antispoof_score'),
        }
    attendance_log.append(log_entry)
    save_attendance_log()

    return jsonify({
        'success':    True,
        'name':       name,
        'confidence': float(confidence),
        'timestamp':  timestamp.strftime('%Y-%m-%d %H:%M:%S'),
    })


@app.route('/get_users', methods=['GET'])
def get_users():
    users = face_recognizer.get_all_users()
    return jsonify({'users': users, 'count': len(users)})

@app.route('/get_logs', methods=['GET'])
def get_logs():
    return jsonify({'logs': attendance_log})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global current_mode
    mode = request.json.get('mode')
    if mode in ['attendance', 'login']:
        current_mode = mode
        return jsonify({'success': True, 'mode': current_mode})
    return jsonify({'success': False, 'message': 'Invalid mode'})

@app.route('/toggle_antispoof', methods=['POST'])
def toggle_antispoof():
    global anti_spoof_enabled
    enabled = request.json.get('enabled')
    if enabled is not None:
        anti_spoof_enabled = bool(enabled)
        return jsonify({'success': True, 'enabled': anti_spoof_enabled})
    return jsonify({'success': False, 'message': 'Invalid parameter'})

@app.route('/get_antispoof_status', methods=['GET'])
def get_antispoof_status():
    return jsonify({
        'enabled': anti_spoof_enabled,
        'method':  'DeepFace FasNet (MiniFASNet V2 + V1SE ensemble)',
    })

@app.route('/debug_scores', methods=['GET'])
def debug_scores():
    return jsonify(last_debug_scores)

@app.route('/shutdown', methods=['POST'])
def shutdown():
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return jsonify({'success': True})


def save_attendance_log():
    os.makedirs('data', exist_ok=True)
    with open('data/attendance_log.json', 'w') as f:
        json.dump(attendance_log, f, indent=2)

def load_attendance_log():
    global attendance_log
    if os.path.exists('data/attendance_log.json'):
        with open('data/attendance_log.json', 'r') as f:
            attendance_log = json.load(f)


if __name__ == '__main__':
    load_attendance_log()

    print("=" * 60)
    print("  Face Recognition + Anti-Spoofing Attendance System")
    print("=" * 60)
    print("  Preloading anti-spoof model (first run downloads ~20 MB)...")
    _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    anti_spoof.predict(_dummy, (100, 100, 200, 200))
    print("  Anti-spoof model ready.")
    print("=" * 60)
    print("  Face detection  : MediaPipe BlazeFace (Tasks API)")
    print("  Face recognition: DeepFace / FaceNet (cosine sim)")
    print("  Anti-spoofing   : DeepFace FasNet — MiniFASNet ensemble")
    print("    - MiniFASNetV2   (2.7x crop, 80x80)")
    print("    - MiniFASNetV1SE (4.0x crop, 80x80)")
    print("    - Passive: no user cooperation required")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)

    # Start background anti-spoof thread AFTER model is warm
    threading.Thread(target=_spoof_worker, daemon=True).start()

    app.run(debug=False, host='0.0.0.0', port=5000)
