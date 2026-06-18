import cv2
import numpy as np
from deepface import DeepFace


class AntiSpoof:
    """
    Passive anti-spoofing via DeepFace FasNet (MiniFASNet V1SE + V2 ensemble).

    DeepFace bundles its own compatible copy of the FasNet architecture and
    downloads the two model weights (~10 MB each) automatically on first use.
    No user cooperation required — works on a single frame.

    Requires: torch  (CPU-only build is fine)
      pip install torch --index-url https://download.pytorch.org/whl/cpu
    """

    def __init__(self):
        self._last = None   # last successful prediction — used as fallback

    def predict(self, frame: np.ndarray, bbox: tuple) -> dict:
        """
        Run anti-spoofing on the full frame.
        DeepFace handles face detection and multi-scale cropping internally.
        """
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = DeepFace.extract_faces(
                img_path=rgb,
                anti_spoofing=True,
                enforce_detection=False,
                detector_backend='opencv',
            )
            if faces:
                face    = faces[0]
                is_real = bool(face['is_real'])
                score   = float(face['antispoof_score'])
                conf    = score if is_real else 1.0 - score
                result = {
                    'is_real':    is_real,
                    'confidence': conf,
                    'label':      'Real' if is_real else 'Spoof',
                    'scores':     {'antispoof_score': score},
                    'details':    {},
                }
                self._last = result
                return result
        except Exception as e:
            print(f"[antispoof] {e}")

        # DeepFace detector missed the face — return last known result so we
        # don't flip to Real on a single bad frame. Fail-secure on first call.
        if self._last is not None:
            return self._last
        return {
            'is_real':    False,
            'confidence': 0.5,
            'label':      'Unknown',
            'scores':     {},
            'details':    {},
        }

    def annotate_frame(self, frame: np.ndarray, bbox: tuple, result: dict) -> np.ndarray:
        x, y, w, h = bbox
        color = (0, 255, 0) if result['is_real'] else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = result['label']
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def get_color_for_prediction(self, result: dict) -> tuple:
        return (0, 255, 0) if result['is_real'] else (0, 0, 255)

    def reset_temporal_state(self):
        pass   # stateless — nothing to reset
