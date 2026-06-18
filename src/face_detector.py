import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


class FaceDetector:
    """
    Face detection using MediaPipe BlazeFace (short-range model).
    Uses the Tasks API introduced in mediapipe >= 0.10.5.
    The TFLite model is downloaded automatically on first run.
    """

    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceDetection = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=self._model_path()),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        self.detector = FaceDetection.create_from_options(options)

    def _model_path(self):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, "blaze_face_short_range.tflite")
        if not os.path.exists(path):
            print("Downloading BlazeFace model...")
            urllib.request.urlretrieve(_MODEL_URL, path)
        return path

    def detect(self, image):
        """Return list of (x, y, w, h) bounding boxes for all detected faces."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        faces = []
        ih, iw = image.shape[:2]
        for det in (results.detections or []):
            bb = det.bounding_box
            x = max(0, bb.origin_x)
            y = max(0, bb.origin_y)
            w = min(bb.width,  iw - x)
            h = min(bb.height, ih - y)
            faces.append((x, y, w, h))
        return faces

    def extract_face(self, image, bbox, target_size=(160, 160)):
        """Crop and resize a face region from the full frame."""
        x, y, w, h = bbox
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return None
        return cv2.resize(crop, target_size)

    def __del__(self):
        if hasattr(self, "detector"):
            self.detector.close()
