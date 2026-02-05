# Face Recognition +  Anti-Spoofing/liveness-detection Attendance System

## 🛡️ Anti-Spoofing Technology

Uses **7 classical computer vision techniques** - NO deep learning required!

1. **LBP** - Texture analysis
2. **HOG** - Edge detection  
3. **DoG** - Frequency analysis
4. **Optical Flow** - Motion detection
5. **Blink Detection** - Liveness check
6. **Color Analysis** - Reflectance patterns
7. **Micro-texture** - Print detection

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python app.py

# Open browser
http://localhost:5000
```

## 📋 Features

✅ Real-time face detection (MediaPipe)
✅ Face recognition (DeepFace)
✅ Anti-spoofing (Classical CV)
✅ Color-coded boxes:
   - 🟢 Green = Real face (>70%)
   - 🟡 Yellow = Real face (50-70%)
   - 🔴 Red = Spoof detected (<50%)

## 🔧 Configuration

Edit `anti_spoof_classical.py` line 48-55:

```python
self.thresholds = {
    'lbp_uniformity': 0.65,    # Lower = stricter
    'edge_density': 0.15,       # Higher = stricter
    'high_freq_ratio': 0.25,    # Higher = stricter
    'flow_variance': 150,       # Higher = stricter
    'color_variance_min': 15,   # Higher = stricter
    'blink_timeout': 5.0        # Blink required in X seconds
}
```

## 📁 Project Structure

```
Face-recognition-AntispoofDetection-AttendanceSystem/
├── app.py                      ← Main application
├── anti_spoof_classical.py     ← Anti-spoofing engine
├── face_detector.py            ← MediaPipe detection
├── face_recognizer.py          ← DeepFace recognition
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    ├── css/
    └── js/
```

## 🐛 Troubleshooting

**Real faces detected as spoof?**
- Increase lighting
- Blink naturally
- Lower thresholds

**Photos detected as real?**
- Increase thresholds
- Better camera quality
- Improve lighting

## 📊 Performance

- Speed: 50ms/frame (20 FPS)
- Accuracy: 85-90%
- Hardware: CPU only
- No GPU required!

