import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import random
import time
import os
from collections import deque

_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = "face_landmarker.task"

CHALLENGES = {
    'blink':      'BLINK your eyes',
    'turn_head':  'TURN your head left or right',
    'open_mouth': 'OPEN your mouth wide',
}

# Face Mesh landmark indices (same topology as mp.solutions.face_mesh)
_LEFT_EYE  = [33,  160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
_NOSE_TIP  = 4
_L_EDGE    = 234   # left face boundary (near left ear)
_R_EDGE    = 454   # right face boundary
_UPPER_LIP = 13
_LOWER_LIP = 14
_L_MOUTH   = 61
_R_MOUTH   = 291


def _download_model():
    if not os.path.exists(_MODEL_PATH):
        print("Downloading Face Landmarker model (~30 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Face Landmarker model downloaded.")


class LivenessChallenge:
    """
    Challenge-response liveness detection via MediaPipe Face Landmarker.

    Three challenges (random per session):
      blink      – Eye Aspect Ratio drops below resting baseline
      turn_head  – Nose position shifts relative to face-width baseline
      open_mouth – Mouth Aspect Ratio rises above resting baseline

    All measures are dimensionless ratios, invariant to face size and
    camera distance.  A one-second warm-up collects the resting baseline
    before the trigger window opens.
    """

    def __init__(self):
        _download_model()
        FaceLandmarker        = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        BaseOptions           = mp.tasks.BaseOptions
        RunningMode           = mp.tasks.vision.RunningMode

        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(opts)
        self._reset()

    # ── public API ───────────────────────────────────────────────────────

    def start(self):
        """Pick a random challenge and start the timer. Returns info dict."""
        self._reset()
        self.challenge   = random.choice(list(CHALLENGES))
        self.instruction = CHALLENGES[self.challenge]
        self.start_time  = time.time()
        return {
            'challenge':   self.challenge,
            'instruction': self.instruction,
            'timeout':     self.timeout,
        }

    def process_frame(self, frame):
        """
        Feed the latest camera frame. Returns a status dict:
          done, success, timed_out, time_remaining, challenge, instruction
        """
        if self.start_time is None:
            return self._status()

        elapsed   = time.time() - self.start_time
        remaining = max(0.0, self.timeout - elapsed)

        if self._done:
            return self._status(remaining)

        if elapsed >= self.timeout:
            self._done = True
            return self._status(0.0, timed_out=True)

        lm = self._get_landmarks(frame)
        if lm is None:
            return self._status(remaining)

        val = self._measure(lm)
        self._buf.append(val)

        # Collect resting baseline during first second
        if self._baseline is None:
            if elapsed >= 1.0 and len(self._buf) >= 3:
                samples = list(self._buf)
                self._baseline = (max(samples)
                                  if self.challenge == 'blink'
                                  else float(np.median(samples)))
            return self._status(remaining)

        if self._triggered(val):
            self._done    = True
            self._success = True

        return self._status(remaining)

    def reset(self):
        self._reset()

    @property
    def active(self):
        return self.start_time is not None and not self._done

    # ── internals ────────────────────────────────────────────────────────

    def _reset(self):
        self.challenge   = None
        self.instruction = None
        self.start_time  = None
        self.timeout     = 6.0
        self._baseline   = None
        self._buf        = deque(maxlen=10)
        self._success    = False
        self._done       = False

    def _get_landmarks(self, frame):
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]   # list of NormalizedLandmark

    def _ear(self, lm, idx):
        """Eye Aspect Ratio from 6 landmark indices."""
        p  = [(lm[i].x, lm[i].y) for i in idx]
        v1 = np.hypot(p[1][0] - p[5][0], p[1][1] - p[5][1])
        v2 = np.hypot(p[2][0] - p[4][0], p[2][1] - p[4][1])
        h  = np.hypot(p[0][0] - p[3][0], p[0][1] - p[3][1])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _measure(self, lm):
        if self.challenge == 'blink':
            l = self._ear(lm, _LEFT_EYE)
            r = self._ear(lm, _RIGHT_EYE)
            return (l + r) / 2.0

        if self.challenge == 'turn_head':
            nx = lm[_NOSE_TIP].x
            lx = lm[_L_EDGE].x
            rx = lm[_R_EDGE].x
            return (nx - lx) / (abs(rx - lx) + 1e-6)   # ~0.5 when centred

        if self.challenge == 'open_mouth':
            vert  = abs(lm[_UPPER_LIP].y - lm[_LOWER_LIP].y)
            horiz = abs(lm[_L_MOUTH].x   - lm[_R_MOUTH].x)
            return vert / (horiz + 1e-6)

        return 0.0

    def _triggered(self, val):
        b = self._baseline
        if self.challenge == 'blink':
            return val < b * 0.62 and val < 0.21

        if self.challenge == 'turn_head':
            return abs(val - b) > 0.17

        if self.challenge == 'open_mouth':
            return val > b + 0.22 and val > 0.30

        return False

    def _status(self, remaining=0.0, timed_out=False):
        return {
            'done':          self._done,
            'success':       self._success,
            'timed_out':     timed_out,
            'time_remaining': round(remaining, 1),
            'challenge':     self.challenge,
            'instruction':   self.instruction,
        }

    def __del__(self):
        if hasattr(self, '_landmarker'):
            self._landmarker.close()
