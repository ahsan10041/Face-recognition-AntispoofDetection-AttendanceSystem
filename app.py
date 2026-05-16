from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from anti_spoof import AntiSpoof
import json
from datetime import datetime
import os

app = Flask(__name__)

# Initialize components
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
anti_spoof = AntiSpoof()

# Global variables
camera = None
current_mode = 'attendance'
attendance_log = []
anti_spoof_enabled = True

def get_camera():
    """Get camera instance, preferring external webcam (index 1) over built-in (index 0)"""
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
    """Generate video frames with face detection and anti-spoofing"""
    while True:
        cam = get_camera()
        success, frame = cam.read()
        
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        faces = face_detector.detect(frame)
        
        for bbox in faces:
            x, y, w, h = bbox
            
            if anti_spoof_enabled:
                spoof_result = anti_spoof.predict(frame, bbox)
                frame = anti_spoof.annotate_frame(frame, bbox, spoof_result)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
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
    
    spoof_result = None
    if anti_spoof_enabled:
        spoof_result = anti_spoof.predict(frame, faces[0])
        if not spoof_result['is_real']:
            return jsonify({
                'success': False,
                'message': f'Spoof detected! ({spoof_result["confidence"]:.0%})',
                'spoof_detected': True,
                'spoof_result': {
                    'label': spoof_result['label'],
                    'confidence': float(spoof_result['confidence']),
                    'scores': {k: float(v) for k, v in spoof_result['scores'].items()}
                }
            })
    
    ret, buffer = cv2.imencode('.jpg', face_img)
    img_str = buffer.tobytes()
    
    response_data = {
        'success': True,
        'face_data': img_str.hex(),
        'bbox': faces[0]
    }
    
    if spoof_result:
        response_data['spoof_result'] = {
            'label': spoof_result['label'],
            'confidence': float(spoof_result['confidence']),
            'scores': {k: float(v) for k, v in spoof_result['scores'].items()}
        }
    
    return jsonify(response_data)

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name', '').strip()
    face_data = data.get('face_data')
    
    if not name:
        return jsonify({'success': False, 'message': 'Name is required'})
    
    if not face_data:
        return jsonify({'success': False, 'message': 'Face data is required'})
    
    face_bytes = bytes.fromhex(face_data)
    face_array = np.frombuffer(face_bytes, dtype=np.uint8)
    face_img = cv2.imdecode(face_array, cv2.IMREAD_COLOR)
    
    success, message = face_recognizer.register_user(name, face_img)
    
    return jsonify({'success': success, 'message': message})

@app.route('/recognize', methods=['POST'])
def recognize():
    cam = get_camera()
    success, frame = cam.read()
    
    if not success:
        return jsonify({'success': False, 'message': 'Failed to capture frame'})
    
    frame = cv2.flip(frame, 1)
    faces = face_detector.detect(frame)
    
    if len(faces) == 0:
        return jsonify({'success': False, 'message': 'No face detected'})
    
    face_img = face_detector.extract_face(frame, faces[0])
    
    if face_img is None:
        return jsonify({'success': False, 'message': 'Failed to extract face'})
    
    spoof_result = None
    if anti_spoof_enabled:
        spoof_result = anti_spoof.predict(frame, faces[0])
        if not spoof_result['is_real']:
            return jsonify({
                'success': False,
                'message': f'Spoof attack detected! ({spoof_result["confidence"]:.0%})',
                'spoof_detected': True
            })
    
    name, confidence = face_recognizer.recognize(face_img)
    
    if name is None:
        return jsonify({'success': False, 'message': 'Unknown user', 'confidence': float(confidence)})
    
    timestamp = datetime.now()
    log_entry = {
        'name': name,
        'timestamp': timestamp.isoformat(),
        'confidence': float(confidence),
        'mode': current_mode
    }
    
    if spoof_result:
        log_entry['anti_spoof'] = {
            'label': spoof_result['label'],
            'confidence': float(spoof_result['confidence'])
        }
    
    attendance_log.append(log_entry)
    save_attendance_log()
    
    response_data = {
        'success': True,
        'name': name,
        'confidence': float(confidence),
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify(response_data)

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
    data = request.json
    mode = data.get('mode')
    
    if mode in ['attendance', 'login']:
        current_mode = mode
        return jsonify({'success': True, 'mode': current_mode})
    
    return jsonify({'success': False, 'message': 'Invalid mode'})

@app.route('/toggle_antispoof', methods=['POST'])
def toggle_antispoof():
    global anti_spoof_enabled
    data = request.json
    enabled = data.get('enabled')
    
    if enabled is not None:
        anti_spoof_enabled = bool(enabled)
        if anti_spoof_enabled:
            anti_spoof.reset_temporal_state()
        return jsonify({'success': True, 'enabled': anti_spoof_enabled})
    
    return jsonify({'success': False, 'message': 'Invalid parameter'})

@app.route('/get_antispoof_status', methods=['GET'])
def get_antispoof_status():
    return jsonify({
        'enabled': anti_spoof_enabled,
        'method': 'Classical CV (LBP + optical flow + colour + FFT)'
    })

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
    print("  Face detection  : MediaPipe BlazeFace")
    print("  Face recognition: DeepFace / FaceNet (cosine sim)")
    print("  Anti-spoofing   : Classical CV (self-implemented)")
    print("    - LBP skin texture entropy")
    print("    - Optical flow liveness")
    print("    - Temporal blood-flow color variation")
    print("    - Screen emission / brightness uniformity")
    print("    - Screen colour saturation + blue-bias")
    print("    - Moire / pixel-grid FFT analysis")
    print("    - Edge sharpness anomaly")
    print("=" * 60)
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    print()
    
    app.run(debug=False, host='0.0.0.0', port=5000)