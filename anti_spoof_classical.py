import cv2
import numpy as np
from collections import deque
from skimage.feature import local_binary_pattern


class AntiSpoofClassical:
    """
    Classical Anti-Spoofing (Presentation Attack Detection)

    REAL  = Live human face
    SPOOF = Screen replay OR printed photo
    """

    def __init__(self):
        self.prev_gray = None
        self.temporal_diff = deque(maxlen=15)
        self.color_history = deque(maxlen=25)

    # --------------------------------------------------
    # 1. Rolling shutter / horizontal banding (screens)
    # --------------------------------------------------
    def detect_banding(self, gray):
        row_means = np.mean(gray, axis=1)
        band_strength = np.std(np.diff(row_means))
        return float(min(band_strength / 2.0, 1.0))

    # --------------------------------------------------
    # 2. Screen emissive light + brightness uniformity
    # --------------------------------------------------
    def detect_emission(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(float)

        # Screens: unusually high ratio of very bright pixels
        bright_ratio = np.sum(v > 225) / v.size
        emission = float(min(bright_ratio * 7.0, 1.0))

        # Screens / flat prints: unnaturally uniform brightness across face region
        v_std = float(np.std(v))
        uniformity = float(np.clip(1.0 - v_std / 35.0, 0.0, 1.0))

        return float(0.55 * emission + 0.45 * uniformity)

    # --------------------------------------------------
    # 3. Optical-flow liveness
    #    Replaces the old simple pixel-diff stability check.
    #    Physical basis:
    #      - Real faces have micro-motion (breathing, micro-expressions, head sway)
    #      - Static prints → near-zero flow
    #      - Screen replay → spatially uniform or looped flow
    #      - Flat object held by hand → rigid 2-D translation (low spatial variance in flow)
    #      - Real 3-D face → non-rigid, higher spatial variance even for small motions
    # --------------------------------------------------
    def detect_motion_liveness(self, gray):
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray.copy()
            return 0.5

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=2, winsize=12,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = gray.copy()

        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        self.temporal_diff.append(mean_mag)

        # Average over recent frames to smooth out camera shake
        avg_motion = float(np.mean(list(self.temporal_diff))) if len(self.temporal_diff) >= 5 else mean_mag

        # Near-zero sustained motion → likely spoof (static print or static screen)
        too_still = float(np.clip(1.0 - avg_motion / 0.55, 0.0, 1.0))

        # Spatially uniform flow → flat object translating rigidly (paper / screen)
        # Real 3-D face has higher spatial variance even for small head movements
        if mean_mag > 0.08:
            spatial_uniformity = float(np.clip(1.0 - std_mag / (mean_mag * 2.0), 0.0, 1.0))
        else:
            spatial_uniformity = 0.75  # very little motion → treat as suspicious

        return float(np.clip(0.60 * too_still + 0.40 * spatial_uniformity, 0.0, 1.0))

    # --------------------------------------------------
    # 4. Moiré / pixel-grid frequency (screens)
    # --------------------------------------------------
    def detect_moire(self, gray):
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        h, w = magnitude.shape
        center = magnitude[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
        ring = magnitude[h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40]
        ratio = np.mean(ring) / (np.mean(center) + 1e-6)
        return float(min(max((ratio - 1.0) * 2.0, 0.0), 1.0))

    # --------------------------------------------------
    # 5. Skin micro-texture loss (LBP entropy)
    #    Threshold raised from 1.5 → 1.8:
    #    A good webcam resolves finer skin detail, pushing real-face LBP
    #    entropy higher (~2.0+). Good-quality prints still land ~1.4-1.7,
    #    so raising the threshold catches them without flagging real faces.
    # --------------------------------------------------
    def detect_texture_loss(self, gray):
        lbp = local_binary_pattern(gray, 8, 1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        entropy = -np.sum(hist * np.log(hist + 1e-6))
        return float(np.clip((1.8 - entropy) / 1.8, 0.0, 1.0))

    # --------------------------------------------------
    # 6. Temporal color variation — blood-flow liveness signal
    #    Physical basis:
    #      - Real skin subtly pulses in the red channel due to blood flow
    #        (~0.5-2 intensity units across 20 frames at 30 fps)
    #      - Printed photos and screen replays are temporally stable in color
    #    Returns high score (spoof) when color is very stable across frames.
    # --------------------------------------------------
    def detect_temporal_color(self, bgr):
        r = float(np.mean(bgr[:, :, 2]))
        g = float(np.mean(bgr[:, :, 1]))
        self.color_history.append((r, g))

        if len(self.color_history) < 8:
            return 0.5

        r_vals = [x[0] for x in self.color_history]
        g_vals = [x[1] for x in self.color_history]
        variation = (float(np.std(r_vals)) + float(np.std(g_vals))) / 2.0

        # variation < 2.5 intensity units → suspiciously stable → spoof
        return float(np.clip(1.0 - variation / 2.5, 0.0, 1.0))

    # --------------------------------------------------
    # 7. Edge sharpness anomaly (printer / compression artifacts)
    # --------------------------------------------------
    def detect_edge_sharpness(self, gray):
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.mean(edges > 0))
        return float(np.clip((edge_density - 0.08) / 0.12, 0.0, 1.0))

    # ==================================================
    # MAIN PREDICTION
    # ==================================================
    def predict(self, face_image, full_frame=None):
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # --- Screen-attack detectors ---
        band   = self.detect_banding(gray)
        emit   = self.detect_emission(face_image)
        motion = self.detect_motion_liveness(gray)
        moire  = self.detect_moire(gray)

        display_score = (
            0.15 * band   +
            0.25 * emit   +
            0.40 * motion +
            0.20 * moire
        )

        # --- Print-attack detectors ---
        texture   = self.detect_texture_loss(gray)
        color_var = self.detect_temporal_color(face_image)
        sharp     = self.detect_edge_sharpness(gray)

        print_score = (
            0.40 * texture   +
            0.35 * color_var +
            0.25 * sharp
        )

        spoof_score = max(display_score, print_score)

        # Threshold lowered 0.45 → 0.40: better spoof sensitivity on a good webcam
        is_real = spoof_score < 0.40
        confidence = 1.0 - spoof_score if is_real else spoof_score

        return {
            'is_real': is_real,
            'confidence': float(confidence),
            'label': 'Real' if is_real else 'Spoof',
            'scores': {
                'banding': float(band),
                'emission': float(emit),
                'motion_liveness': float(motion),
                'moire': float(moire),
                'texture_loss': float(texture),
                'color_variation': float(color_var),
                'edge_sharpness': float(sharp),
                'display_score': float(display_score),
                'print_score': float(print_score),
                'spoof_score': float(spoof_score)
            },
            'details': {}
        }

    # --------------------------------------------------
    # Visualization helpers
    # --------------------------------------------------
    def get_color_for_prediction(self, result):
        return (0, 255, 0) if result['is_real'] else (0, 0, 255)

    def annotate_frame(self, frame, bbox, result):
        x, y, w, h = bbox
        color = self.get_color_for_prediction(result)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{result['label']} ({int(result['confidence'] * 100)}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def reset_temporal_state(self):
        self.prev_gray = None
        self.temporal_diff.clear()
        self.color_history.clear()
