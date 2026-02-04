import cv2
import numpy as np
from collections import deque
from skimage.feature import local_binary_pattern


class AntiSpoofClassical:
    """
    Classical Anti-Spoofing (Presentation Attack Detection)

    REAL   = Live human face
    SPOOF  = Screen replay OR printed photo
    """

    def __init__(self):
        self.prev_gray = None
        self.temporal_diff = deque(maxlen=10)

    # --------------------------------------------------
    # 1. Rolling shutter / horizontal banding (screens)
    # --------------------------------------------------
    def detect_banding(self, gray):
        row_means = np.mean(gray, axis=1)
        band_strength = np.std(np.diff(row_means))
        score = min(band_strength / 2.0, 1.0)
        return score

    # --------------------------------------------------
    # 2. Emissive light detection (screens emit light)
    # --------------------------------------------------
    def detect_emission(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        saturated_ratio = np.sum(v > 240) / v.size
        score = min(saturated_ratio * 8.0, 1.0)
        return score

    # --------------------------------------------------
    # 3. Temporal stability (screens & prints are stable)
    # --------------------------------------------------
    def detect_stability(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return 0.5

        diff = cv2.absdiff(self.prev_gray, gray)
        noise = np.mean(diff)
        self.temporal_diff.append(noise)

        avg_noise = np.mean(self.temporal_diff)
        self.prev_gray = gray.copy()

        score = max(0.0, 1.0 - avg_noise / 12.0)
        return score

    # --------------------------------------------------
    # 4. Moiré / pixel grid frequency (screens)
    # --------------------------------------------------
    def detect_moire(self, gray):
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        h, w = magnitude.shape
        center = magnitude[h//2-10:h//2+10, w//2-10:w//2+10]
        ring = magnitude[h//2-40:h//2+40, w//2-40:w//2+40]

        ratio = np.mean(ring) / (np.mean(center) + 1e-6)
        score = min(max((ratio - 1.0) * 2.0, 0.0), 1.0)
        return score

    # ==================================================
    # PRINT ATTACK DETECTORS (NEW, NON-BREAKING)
    # ==================================================

    # --------------------------------------------------
    # 5. Skin micro-texture loss (LBP entropy)
    # --------------------------------------------------
    def detect_texture_loss(self, gray):
        radius = 1
        points = 8 * radius

        lbp = local_binary_pattern(gray, points, radius, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, points + 3),
            density=True
        )

        entropy = -np.sum(hist * np.log(hist + 1e-6))

        # Printed faces → low entropy
        score = np.clip((1.5 - entropy) / 1.5, 0.0, 1.0)
        return score

    # --------------------------------------------------
    # 6. Planarity / flatness detection (printed photos)
    # --------------------------------------------------
    def detect_planarity(self, gray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        grad_mag = np.sqrt(gx**2 + gy**2)
        variance = np.var(grad_mag)

        # Printed images → very uniform gradients
        score = np.clip((20.0 - variance) / 20.0, 0.0, 1.0)
        return score

    # --------------------------------------------------
    # 7. Edge sharpness anomaly (printer artifacts)
    # --------------------------------------------------
    def detect_edge_sharpness(self, gray):
        edges = cv2.Canny(gray, 80, 160)
        edge_density = np.mean(edges > 0)

        score = np.clip((edge_density - 0.08) / 0.12, 0.0, 1.0)
        return score

    # ==================================================
    # MAIN PREDICTION (UNCHANGED API)
    # ==================================================
    def predict(self, face_image, full_frame=None):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # Screen-based cues
        band = self.detect_banding(gray)
        emit = self.detect_emission(face_image)
        stab = self.detect_stability(gray)
        moire = self.detect_moire(gray)

        display_score = (
            0.30 * band +
            0.30 * emit +
            0.25 * stab +
            0.15 * moire
        )

        # Print-based cues
        texture = self.detect_texture_loss(gray)
        planar = self.detect_planarity(gray)
        sharp = self.detect_edge_sharpness(gray)

        print_score = (
            0.40 * texture +
            0.35 * planar +
            0.25 * sharp
        )

        # Final spoof decision
        spoof_score = max(display_score, print_score)

        is_real = spoof_score < 0.45
        confidence = 1.0 - spoof_score if is_real else spoof_score

        return {
            'is_real': is_real,
            'confidence': float(confidence),
            'label': 'Real' if is_real else 'Spoof',
            'scores': {
                'banding': float(band),
                'emission': float(emit),
                'stability': float(stab),
                'moire': float(moire),
                'texture_loss': float(texture),
                'planarity': float(planar),
                'edge_sharpness': float(sharp),
                'display_score': float(display_score),
                'print_score': float(print_score),
                'spoof_score': float(spoof_score)
            },
            'details': {}
        }

    # --------------------------------------------------
    # Visualization helpers (UNCHANGED)
    # --------------------------------------------------
    def get_color_for_prediction(self, result):
        return (0, 255, 0) if result['is_real'] else (0, 0, 255)

    def annotate_frame(self, frame, bbox, result):
        x, y, w, h = bbox
        color = self.get_color_for_prediction(result)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"{result['label']} ({int(result['confidence'] * 100)}%)"
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        return frame

    def reset_temporal_state(self):
        self.prev_gray = None
        self.temporal_diff.clear()
