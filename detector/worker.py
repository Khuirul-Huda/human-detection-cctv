"""
Worker thread managing the detection pipeline for an individual camera stream.
"""

import time
import logging
import threading
from typing import Optional
import numpy as np

from .config import AppConfig, CameraConfig
from .stream import RTSPFFmpegStream
from .motion import MotionGate
from .inference import SharedYOLODetector
from .notifier import TelegramNotifier

logger = logging.getLogger("CCTV-Detector.Worker")


class CameraWorker:
    """Orchestrates stream reading, motion evaluation, AI inference, and alerts for one camera."""

    def __init__(
        self,
        config: CameraConfig,
        app_config: AppConfig,
        detector: Optional[SharedYOLODetector],
        notifier: TelegramNotifier,
        enable_ai: bool = True,
    ):
        self.config = config
        self.app_config = app_config
        self.detector = detector
        self.notifier = notifier
        self.enable_ai = enable_ai

        self.stream = RTSPFFmpegStream(
            name=config.name,
            url=config.url,
            transport=config.transport,
            width=app_config.stream_width,
            height=app_config.stream_height,
            fps=app_config.stream_fps,
            threads=app_config.ff_threads,
        )
        self.motion_gate = MotionGate(
            threshold=app_config.motion_threshold,
            delta_thresh=app_config.motion_delta_threshold,
            blur_size=app_config.motion_blur_size,
            dilate_iter=app_config.motion_dilate_iterations,
            alpha=app_config.motion_alpha,
            warmup_frames=app_config.motion_warmup_frames,
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.last_ai_time = 0.0
        self.motion_events = 0
        self.human_events = 0

    def start(self) -> None:
        """Starts stream reader and camera processing loop."""
        self._running = True
        self.stream.start()
        self._thread = threading.Thread(target=self._loop, name=f"Worker-{self.config.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops stream reader and waits for worker loop to finish."""
        self._running = False
        self.stream.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        logger.info(f"[{self.config.name}] Worker pipeline started.")
        last_log_time = time.time()

        while self._running:
            has_frame, frame = self.stream.get_frame()
            if not has_frame or frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            is_motion, delta_pixels = self.motion_gate.check_motion(frame)

            # Periodic heartbeat log every 15 seconds when idle
            if (now - last_log_time) >= 15.0:
                last_log_time = now
                logger.info(
                    f"[{self.config.name}] Status: idle (last delta={delta_pixels}px, thresh={self.app_config.motion_threshold})"
                )

            if is_motion:
                self.motion_events += 1
                logger.info(
                    f"[{self.config.name}] MOTION DETECTED (delta={delta_pixels}px >= {self.app_config.motion_threshold})"
                )

                # Throttle AI calls
                if self.enable_ai and self.detector and (now - self.last_ai_time) >= self.app_config.ai_min_interval:
                    self.last_ai_time = now
                    has_human, highest_conf, boxes = self.detector.detect_human(frame)

                    if has_human:
                        self.human_events += 1
                        logger.info(
                            f"[{self.config.name}] HUMAN CONFIRMED! (conf={highest_conf:.2f}, count={len(boxes)})"
                        )
                        self.notifier.send_alert(self.config.name, highest_conf, frame, boxes)
                    else:
                        logger.info(f"[{self.config.name}] AI Checked: Motion is not a human (filtered).")
