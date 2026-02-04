import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
    
    def detect(self, image):
        """
        Detect faces in image
        Returns: list of (x, y, w, h) bounding boxes
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                
                x = max(0, int(bboxC.xmin * iw))
                y = max(0, int(bboxC.ymin * ih))
                w = min(int(bboxC.width * iw), iw - x)
                h = min(int(bboxC.height * ih), ih - y)
                
                faces.append((x, y, w, h))
        
        return faces
    
    def extract_face(self, image, bbox, target_size=(160, 160)):
        """Extract and resize face region"""
        x, y, w, h = bbox
        face = image[y:y+h, x:x+w]
        
        if face.size == 0:
            return None
        
        face_resized = cv2.resize(face, target_size)
        return face_resized
    
    def __del__(self):
        self.face_detection.close()