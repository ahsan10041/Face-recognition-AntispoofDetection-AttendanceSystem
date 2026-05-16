import cv2
import numpy as np
from collections import deque
from skimage.feature import local_binary_pattern


class AntiSpoof:
    """
    Classical CV anti-spoofing (self-implemented).
    Uses texture, motion, frequency, color and emission cues.
    No external model required.
    """

    def __init__(self):
        self.prev_gray = None
        self.temporal_diff = deque(maxlen=15)
        self.color_history = deque(maxlen=25)

    # ------------------------------------------------------------------
    # Face crop helper
    # ------------------------------------------------------------------
    def _crop(self, frame, bbox, scale=1.3):
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        nw, nh = int(w * scale), int(h * scale)
        x1 = max(0, cx - nw // 2)
        y1 = max(0, cy - nh // 2)
        x2 = min(frame.shape[1], x1 + nw)
        y2 = min(frame.shape[0], y1 + nh)
        return frame[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # 1. Rolling shutter / horizontal banding (screens)
    # ------------------------------------------------------------------
    def detect_banding(self, gray):
        row_means = np.mean(gray, axis=1)
        band_strength = np.std(np.diff(row_means))
        return float(min(band_strength / 2.0, 1.0))

    # ------------------------------------------------------------------
    # 2. Emissive light + brightness uniformity
    # ------------------------------------------------------------------
    def detect_emission(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(float)
        bright_ratio = np.sum(v > 200) / v.size          # lowered from 225
        emission = float(min(bright_ratio * 6.0, 1.0))
        v_std = float(np.std(v))
        uniformity = float(np.clip(1.0 - v_std / 40.0, 0.0, 1.0))
        return float(0.55 * emission + 0.45 * uniformity)

    # ------------------------------------------------------------------
    # 3. Screen color: over-saturation + blue bias
    #    Phone screens display skin with artificially high S values
    #    and a slight blue-channel bias from the LED backlight.
    # ------------------------------------------------------------------
    def detect_screen_color(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].astype(float) / 255.0
        mean_sat = float(np.mean(s))
        # Real skin typically mean S ≈ 0.35-0.55; screens push it higher
        over_sat = float(np.clip((mean_sat - 0.42) / 0.28, 0.0, 1.0))

        # Blue/Red channel ratio: phone backlights are cooler than skin
        b_mean = float(np.mean(bgr[:, :, 0]))
        r_mean = float(np.mean(bgr[:, :, 2])) + 1e-6
        blue_bias = float(np.clip((b_mean / r_mean - 0.65) / 0.35, 0.0, 1.0))

        return float(0.65 * over_sat + 0.35 * blue_bias)

    # ------------------------------------------------------------------
    # 4. Optical-flow liveness
    # ------------------------------------------------------------------
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
        std_mag  = float(np.std(mag))
        self.temporal_diff.append(mean_mag)

        avg_motion = float(np.mean(list(self.temporal_diff))) if len(self.temporal_diff) >= 5 else mean_mag
        too_still = float(np.clip(1.0 - avg_motion / 0.55, 0.0, 1.0))

        if mean_mag > 0.08:
            spatial_uniformity = float(np.clip(1.0 - std_mag / (mean_mag * 2.0), 0.0, 1.0))
        else:
            spatial_uniformity = 0.75

        return float(np.clip(0.60 * too_still + 0.40 * spatial_uniformity, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 5. Moiré / pixel-grid frequency (screens)
    # ------------------------------------------------------------------
    def detect_moire(self, gray):
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)
        h, w = magnitude.shape
        center = magnitude[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10]
        ring   = magnitude[h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40]
        ratio  = np.mean(ring) / (np.mean(center) + 1e-6)
        return float(min(max((ratio - 1.0) * 2.0, 0.0), 1.0))

    # ------------------------------------------------------------------
    # 6. Skin micro-texture (LBP entropy)
    # ------------------------------------------------------------------
    def detect_texture_loss(self, gray):
        lbp = local_binary_pattern(gray, 8, 1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        entropy = -np.sum(hist * np.log(hist + 1e-6))
        return float(np.clip((1.8 - entropy) / 1.8, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 7. Temporal color variation (blood-flow liveness)
    # ------------------------------------------------------------------
    def detect_temporal_color(self, bgr):
        r = float(np.mean(bgr[:, :, 2]))
        g = float(np.mean(bgr[:, :, 1]))
        self.color_history.append((r, g))
        if len(self.color_history) < 8:
            return 0.5
        r_vals = [x[0] for x in self.color_history]
        g_vals = [x[1] for x in self.color_history]
        variation = (float(np.std(r_vals)) + float(np.std(g_vals))) / 2.0
        return float(np.clip(1.0 - variation / 2.5, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 8. Edge sharpness (compression / print artifacts)
    # ------------------------------------------------------------------
    def detect_edge_sharpness(self, gray):
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.mean(edges > 0))
        return float(np.clip((edge_density - 0.08) / 0.12, 0.0, 1.0))

    # ==================================================================
    # MAIN PREDICTION  (frame + bbox API, matches MiniFASNet wrapper)
    # ==================================================================
    def predict(self, frame: np.ndarray, bbox: tuple) -> dict:
        face = self._crop(frame, bbox, scale=1.3)
        if face.size == 0:
            return {"is_real": True, "confidence": 0.5, "label": "Unknown",
                    "scores": {}, "details": {}}

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # --- Screen-attack detectors ---
        band      = self.detect_banding(gray)
        emit      = self.detect_emission(face)
        screen_c  = self.detect_screen_color(face)
        motion    = self.detect_motion_liveness(gray)
        moire     = self.detect_moire(gray)

        display_score = (
            0.10 * band     +
            0.20 * emit     +
            0.25 * screen_c +  # new: color-saturation/blue-bias screen signal
            0.30 * motion   +
            0.15 * moire
        )

        # --- Print-attack detectors ---
        texture   = self.detect_texture_loss(gray)
        color_var = self.detect_temporal_color(face)
        sharp     = self.detect_edge_sharpness(gray)

        print_score = (
            0.40 * texture   +
            0.35 * color_var +
            0.25 * sharp
        )

        spoof_score = max(display_score, print_score)

        # Threshold 0.35: phone screen was scoring 0.39 at 0.40 threshold
        is_real    = spoof_score < 0.35
        confidence = 1.0 - spoof_score if is_real else spoof_score

        return {
            "is_real":    is_real,
            "confidence": float(confidence),
            "label":      "Real" if is_real else "Spoof",
            "scores": {
                "banding":         float(band),
                "emission":        float(emit),
                "screen_color":    float(screen_c),
                "motion_liveness": float(motion),
                "moire":           float(moire),
                "texture_loss":    float(texture),
                "color_variation": float(color_var),
                "edge_sharpness":  float(sharp),
                "display_score":   float(display_score),
                "print_score":     float(print_score),
                "spoof_score":     float(spoof_score),
            },
            "details": {},
        }

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def get_color_for_prediction(self, result: dict) -> tuple:
        return (0, 255, 0) if result["is_real"] else (0, 0, 255)

    def annotate_frame(self, frame: np.ndarray, bbox: tuple, result: dict) -> np.ndarray:
        x, y, w, h = bbox
        color = self.get_color_for_prediction(result)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{result['label']} ({int(result['confidence'] * 100)}%)"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def reset_temporal_state(self):
        self.prev_gray = None
        self.temporal_diff.clear()
        self.color_history.clear()
