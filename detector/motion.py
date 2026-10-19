"""
Lightweight motion gating using moving-average background subtraction.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class MotionGate:
    """Detects pixel differences against an accumulated weighted background."""

    def __init__(
        self,
        threshold: int = 400,
        delta_thresh: int = 18,
        blur_size: int = 15,
        dilate_iter: int = 1,
        alpha: float = 0.3,
        warmup_frames: int = 8,
    ):
        self.threshold = threshold
        self.delta_thresh = delta_thresh
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.dilate_iter = dilate_iter
        self.alpha = alpha
        self.warmup_frames = warmup_frames
        self.frame_count = 0
        self.avg_frame: Optional[np.ndarray] = None

    def check_motion(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Evaluates frame for significant motion.
        Returns (is_motion, delta_pixel_count).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        if self.avg_frame is None:
            self.avg_frame = gray.astype("float")
            self.frame_count = 1
            return False, 0

        self.frame_count += 1
        cv2.accumulateWeighted(gray, self.avg_frame, self.alpha)

        # Ignore initial frames during camera exposure / handshake warmup
        if self.frame_count <= self.warmup_frames:
            return False, 0

        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg_frame))
        thresh = cv2.threshold(frame_delta, self.delta_thresh, 255, cv2.THRESH_BINARY)[1]
        if self.dilate_iter > 0:
            thresh = cv2.dilate(thresh, None, iterations=self.dilate_iter)

        motion_pixels = cv2.countNonZero(thresh)
        is_motion = motion_pixels >= self.threshold
        return is_motion, motion_pixels
