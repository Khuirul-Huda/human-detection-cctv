#!/usr/bin/env python3
"""
Lightweight Multi-Stream Human Detection for CCTV.
Main Application Entry Point.
"""

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
import threading
from typing import List
import cv2
import numpy as np

from detector import (
    AppConfig,
    CameraConfig,
    CameraWorker,
    SharedYOLODetector,
    TelegramNotifier,
    AudioAlerter,
    load_config,
)

# --- Global Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CCTV-Detector")


def run_snapshot_test(cameras: List[CameraConfig], config: AppConfig) -> None:
    """Captures a test snapshot from each selected camera and exits."""
    logger.info("Running snapshot test mode...")
    for c in cameras:
        out_file = f"/tmp/snapshot_{c.name}.jpg"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", c.transport,
            "-i", c.url,
            "-vf", f"scale={config.stream_width}:{config.stream_height}:flags=fast_bilinear",
            "-vframes", "1",
            out_file,
        ]
        rc = subprocess.call(cmd)
        if rc == 0 and os.path.exists(out_file):
            logger.info(f"[{c.name}] Snapshot success -> {out_file} ({os.path.getsize(out_file)} bytes)")
        else:
            logger.error(f"[{c.name}] Snapshot failed!")


def run_telegram_test(cameras: List[CameraConfig], config: AppConfig, notifier: TelegramNotifier) -> None:
    """Captures a snapshot and sends an annotated test alert to Telegram."""
    logger.info("Sending test Telegram alert with snapshot...")
    if not notifier.active:
        logger.error("Telegram notifier is not active. Check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env.")
        return

    c = cameras[0]
    out_file = f"/tmp/snapshot_{c.name}.jpg"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", c.transport,
        "-i", c.url,
        "-vf", f"scale={config.stream_width}:{config.stream_height}:flags=fast_bilinear",
        "-vframes", "1",
        out_file,
    ]
    subprocess.call(cmd)

    test_frame = cv2.imread(out_file) if os.path.exists(out_file) else np.zeros(
        (config.stream_height, config.stream_width, 3), dtype=np.uint8
    )
    fake_boxes = [([50, 50, 100, 150], 0.95)]
    notifier.send_alert(c.name, 0.95, test_frame, fake_boxes)
    time.sleep(3.0)
    logger.info("Test Telegram alert process triggered.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight Multi-Stream CCTV Human Detection")
    parser.add_argument("--no-ai", action="store_true", help="Run motion gating only without AI inference")
    parser.add_argument("--camera", type=str, choices=["kamera1", "kamera2", "all"], default="all", help="Camera filter")
    parser.add_argument("--test-snapshot", action="store_true", help="Capture 1 snapshot from each camera and exit")
    parser.add_argument("--test-telegram", action="store_true", help="Send a test notification to Telegram and exit")
    parser.add_argument("--test-sound", action="store_true", help="Play test chime sound and exit")
    parser.add_argument("--duration", type=int, default=0, help="Run for N seconds then exit (0 = run indefinitely)")
    args = parser.parse_args()

    config = load_config()

    # Diagnostic test modes
    if args.test_sound:
        logger.info(f"Playing test audio chime from: {config.audio_alert_file}...")
        alerter = AudioAlerter(
            sound_file=config.audio_alert_file,
            cooldown_seconds=0,
            enabled=True,
        )
        alerter.play()
        time.sleep(1.0)
        return

    if not config.cameras:
        logger.error("No cameras configured in .env! Please define CAM1_* or CAM2_*.")
        sys.exit(1)

    selected_cameras = config.cameras
    if args.camera != "all":
        selected_cameras = [c for c in config.cameras if c.name == args.camera]

    logger.info(f"Active cameras: {[c.name for c in selected_cameras]}")

    notifier = TelegramNotifier(
        token=config.telegram_token,
        chat_id=config.telegram_chat_id,
        cooldown_seconds=config.alert_cooldown_seconds,
    )

    # Diagnostic test modes
    if args.test_snapshot:
        run_snapshot_test(selected_cameras, config)
        return

    if args.test_telegram:
        run_telegram_test(selected_cameras, config, notifier)
        return

    # Initialize shared AI detector if enabled
    detector = None
    if not args.no_ai:
        try:
            detector = SharedYOLODetector(config.model_path, config.prob_threshold)
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")
            sys.exit(1)
    else:
        logger.info("Running in NO-AI mode (Motion Gate evaluation only).")

    if notifier.active:
        logger.info(f"Telegram alerting active for chat ID {notifier.chat_id}.")
    else:
        logger.warning("Telegram alerting inactive (token or chat_id empty).")

    audio_alerter = AudioAlerter(
        sound_file=config.audio_alert_file,
        cooldown_seconds=config.audio_cooldown_seconds,
        enabled=config.enable_audio_alert,
    )
    if audio_alerter.enabled:
        logger.info(f"Audio alert active using chime: {os.path.basename(config.audio_alert_file)} (cooldown {config.audio_cooldown_seconds}s)")
    else:
        logger.info("Audio alert is disabled in configuration.")

    # Start workers for each selected camera
    workers = []
    for c in selected_cameras:
        w = CameraWorker(
            config=c,
            app_config=config,
            detector=detector,
            notifier=notifier,
            audio_alerter=audio_alerter,
            enable_ai=(not args.no_ai),
        )
        workers.append(w)
        w.start()

    logger.info("All camera pipelines active. Press Ctrl+C to terminate.")

    # Graceful shutdown handler
    stop_event = threading.Event()

    def _sig_handler(sig, frame):
        logger.info("Termination signal received. Shutting down gracefully...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    start_time = time.time()
    try:
        while not stop_event.is_set():
            time.sleep(1.0)
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                logger.info(f"Duration limit of {args.duration}s reached. Stopping...")
                break
    finally:
        for w in workers:
            w.stop()
        logger.info("CCTV Detector stopped cleanly.")


if __name__ == "__main__":
    main()
