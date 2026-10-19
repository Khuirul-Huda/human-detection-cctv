#!/usr/bin/env python3
"""
Lightweight Multi-Stream Human Detection for CCTV.
Branch: feature/lightweight-multistream
"""

import os
import sys
import time
import select
import signal
import shlex
import logging
import argparse
import subprocess
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import onnxruntime as ort

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CCTV-Detector")


def load_env(env_path: str = ".env"):
    """Load key-value pairs from .env without overwriting existing environment."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        logger.warning(f"Error loading {env_path}: {e}")


load_env()

# --- Config Constants ---
STREAM_FPS = int(os.getenv("STREAM_FPS", "1"))
STREAM_WIDTH = int(os.getenv("STREAM_WIDTH", "320"))
STREAM_HEIGHT = int(os.getenv("STREAM_HEIGHT", "320"))
FF_THREADS = int(os.getenv("FF_THREADS", "1"))

MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.onnx")
PROB_THRESHOLD = float(os.getenv("PROB_THRESHOLD", "0.5"))
AI_MIN_INTERVAL = float(os.getenv("AI_MIN_INTERVAL", "3.0"))

MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "400"))
MOTION_DELTA_THRESHOLD = int(os.getenv("MOTION_DELTA_THRESHOLD", "18"))
MOTION_BLUR_SIZE = int(os.getenv("MOTION_BLUR_SIZE", "15"))
MOTION_DILATE_ITERATIONS = int(os.getenv("MOTION_DILATE_ITERATIONS", "1"))
MOTION_ALPHA = float(os.getenv("MOTION_ALPHA", "0.3"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "60"))


class CameraConfig:
    def __init__(self, name: str, url: str, transport: str = "tcp", enabled: bool = True):
        self.name = name
        self.url = url
        self.transport = transport
        self.enabled = enabled


def get_configured_cameras() -> List[CameraConfig]:
    """Parse configured cameras from environment variables."""
    cameras = []
    # Check CAM1 and CAM2
    for i in (1, 2):
        prefix = f"CAM{i}_"
        enabled = str(os.getenv(f"{prefix}ENABLE", "false")).lower() in ("1", "true", "yes")
        name = os.getenv(f"{prefix}NAME", f"kamera{i}")
        url = os.getenv(f"{prefix}URL", "")
        transport = os.getenv(f"{prefix}TRANSPORT", "tcp")
        if enabled and url:
            cameras.append(CameraConfig(name=name, url=url, transport=transport, enabled=True))

    # Fallback to legacy single stream if no CAM1/CAM2
    if not cameras and os.getenv("STREAM_URL"):
        cameras.append(CameraConfig("kamera1", os.getenv("STREAM_URL"), "tcp", True))
    return cameras


class RTSPFFmpegStream:
    """Spawns an FFmpeg worker to read an RTSP stream and outputs raw BGR frames."""

    def __init__(
        self,
        name: str,
        url: str,
        transport: str = "tcp",
        width: int = 320,
        height: int = 320,
        fps: int = 1,
        threads: int = 1,
    ):
        self.name = name
        self.url = url
        self.transport = transport
        self.width = width
        self.height = height
        self.fps = fps
        self.threads = threads
        self.frame_size = width * height * 3

        self.proc: Optional[subprocess.Popen] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._has_new_frame = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name=f"RTSP-{self.name}", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _start_ffmpeg(self) -> Optional[subprocess.Popen]:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-rtsp_transport", self.transport,
            "-fflags", "+genpts+discardcorrupt+nobuffer",
            "-flags", "low_delay",
            "-timeout", "5000000",
            "-threads", str(self.threads),
            "-i", self.url,
            "-vf", f"fps={self.fps},scale={self.width}:{self.height}:flags=fast_bilinear",
            "-pix_fmt", "bgr24",
            "-an",
            "-f", "rawvideo",
            "pipe:1",
        ]
        try:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to start FFmpeg: {e}")
            return None

    def _read_exact(self, num_bytes: int, timeout: float = 6.0) -> Optional[bytes]:
        buf = bytearray()
        last_read_time = time.time()
        while len(buf) < num_bytes and self._running and self.proc and self.proc.poll() is None:
            rlist, _, _ = select.select([self.proc.stdout], [], [], 1.0)
            if not rlist:
                if (time.time() - last_read_time) >= timeout:
                    logger.warning(f"[{self.name}] Stream data timeout ({timeout}s without data). Reconnecting...")
                    return None
                continue
            needed = num_bytes - len(buf)
            chunk = self.proc.stdout.read(min(needed, 65536))
            if not chunk:
                return None
            buf.extend(chunk)
            last_read_time = time.time()
        return bytes(buf) if len(buf) == num_bytes else None

    def _run_loop(self):
        logger.info(f"[{self.name}] Stream reader started ({self.transport.upper()}, {self.fps} FPS, {self.width}x{self.height})")
        while self._running:
            self.proc = self._start_ffmpeg()
            if not self.proc:
                time.sleep(3)
                continue

            try:
                while self._running and self.proc.poll() is None:
                    raw_bytes = self._read_exact(self.frame_size)
                    if not raw_bytes:
                        break

                    frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((self.height, self.width, 3))
                    with self._lock:
                        self._latest_frame = frame
                        self._has_new_frame = True

            except Exception as e:
                logger.warning(f"[{self.name}] Stream read error: {e}")

            if self.proc:
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.proc = None

            if self._running:
                logger.warning(f"[{self.name}] Stream lost. Reconnecting in 3s...")
                time.sleep(3)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns (has_new_frame, frame_copy)."""
        with self._lock:
            if self._has_new_frame and self._latest_frame is not None:
                self._has_new_frame = False
                return True, self._latest_frame.copy()
            return False, None


class MotionGate:
    """Lightweight moving average background subtraction."""

    def __init__(
        self,
        threshold: int = MOTION_THRESHOLD,
        delta_thresh: int = MOTION_DELTA_THRESHOLD,
        blur_size: int = MOTION_BLUR_SIZE,
        dilate_iter: int = MOTION_DILATE_ITERATIONS,
        alpha: float = MOTION_ALPHA,
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
        """Returns (is_motion, delta_pixel_count)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        if self.avg_frame is None:
            self.avg_frame = gray.astype("float")
            self.frame_count = 1
            return False, 0

        self.frame_count += 1
        cv2.accumulateWeighted(gray, self.avg_frame, self.alpha)

        # Skip motion check during initial warmup
        if self.frame_count <= self.warmup_frames:
            return False, 0

        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg_frame))
        thresh = cv2.threshold(frame_delta, self.delta_thresh, 255, cv2.THRESH_BINARY)[1]
        if self.dilate_iter > 0:
            thresh = cv2.dilate(thresh, None, iterations=self.dilate_iter)

        motion_pixels = cv2.countNonZero(thresh)
        is_motion = motion_pixels >= self.threshold
        return is_motion, motion_pixels


class SharedYOLODetector:
    """Thread-safe single instance YOLOv8n ONNX Detector."""

    def __init__(self, model_path: str = MODEL_PATH, prob_threshold: float = PROB_THRESHOLD):
        self.model_path = model_path
        self.prob_threshold = prob_threshold
        self._lock = threading.Lock()
        self.session: Optional[ort.InferenceSession] = None
        self.input_name = ""
        self.output_name = ""
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        logger.info(f"Loading ONNX model: {self.model_path} (Single Core CPU)...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info("YOLOv8n ONNX Model loaded successfully.")

    def detect_human(self, frame: np.ndarray) -> Tuple[bool, float, List[Tuple[List[int], float]]]:
        """
        Detects class 0 (person).
        Returns: (has_person, highest_conf, [(box, conf)])
        box format: [x, y, w, h] in frame coordinates.
        """
        h, w = frame.shape[:2]
        # Preprocessing to 640x640 RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        img_data = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
        img_data = np.expand_dims(img_data, axis=0)

        with self._lock:
            t0 = time.time()
            outputs = self.session.run([self.output_name], {self.input_name: img_data})
            inference_ms = (time.time() - t0) * 1000

        output = outputs[0][0].transpose()  # shape: [8400, 84]

        boxes = []
        confidences = []
        x_factor = w / 640.0
        y_factor = h / 640.0

        for row in output:
            scores = row[4:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            # class_id 0 = person
            if class_id == 0 and conf >= self.prob_threshold:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                width = int(bw * x_factor)
                height = int(bh * y_factor)
                boxes.append([x, y, width, height])
                confidences.append(float(conf))

        if not boxes:
            return False, 0.0, []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.prob_threshold, 0.45)
        final_boxes = []
        highest_conf = 0.0

        if len(indices) > 0:
            for idx in indices.flatten():
                box = boxes[idx]
                conf = confidences[idx]
                final_boxes.append((box, conf))
                if conf > highest_conf:
                    highest_conf = conf
            return True, highest_conf, final_boxes

        return False, 0.0, []


class TelegramNotifier:
    """Asynchronous Telegram alert sender with per-camera cooldown."""

    def __init__(self, token: str = TELEGRAM_TOKEN, chat_id: str = TELEGRAM_CHAT_ID, cooldown: int = ALERT_COOLDOWN_SECONDS):
        self.token = token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self.last_alert_time: Dict[str, float] = {}
        self.active = bool(token and token != "TOKEN_BOT_MU" and chat_id)

    def can_alert(self, cam_name: str) -> bool:
        now = time.time()
        last = self.last_alert_time.get(cam_name, 0.0)
        return (now - last) >= self.cooldown

    def send_alert(self, cam_name: str, highest_conf: float, frame: np.ndarray, boxes: List[Tuple[List[int], float]]):
        if not self.active:
            logger.info(f"[{cam_name}] ALERT: Human detected (conf={highest_conf:.2f}), Telegram not configured.")
            return

        if not self.can_alert(cam_name):
            logger.info(f"[{cam_name}] Human detected (conf={highest_conf:.2f}), alert skipped due to cooldown.")
            return

        self.last_alert_time[cam_name] = time.time()

        # Draw bounding boxes on snapshot
        vis_frame = frame.copy()
        for box, conf in boxes:
            x, y, w, h = box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            label = f"Person {int(conf * 100)}%"
            cv2.putText(vis_frame, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        temp_img_path = f"/tmp/alert_{cam_name}_{int(time.time())}.jpg"
        cv2.imwrite(temp_img_path, vis_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

        text = (
            f"🚨 *PERINGATAN DETEKSI CCTV*\n"
            f"📸 *Kamera:* `{cam_name}`\n"
            f"👤 *Objek:* Manusia Terdeteksi\n"
            f"📊 *Confidence:* `{int(highest_conf * 100)}%`\n"
            f"⏰ *Waktu:* `{time.strftime('%Y-%m-%d %H:%M:%S')}`"
        )

        def _do_send():
            try:
                with open(temp_img_path, "rb") as photo:
                    resp = requests.post(
                        f"https://api.telegram.org/bot{self.token}/sendPhoto",
                        data={"chat_id": self.chat_id, "caption": text, "parse_mode": "Markdown"},
                        files={"photo": photo},
                        timeout=12,
                    )
                if resp.status_code == 200:
                    logger.info(f"[{cam_name}] Telegram alert photo sent successfully.")
                else:
                    logger.warning(f"[{cam_name}] Telegram send failed: {resp.text}")
            except Exception as e:
                logger.warning(f"[{cam_name}] Failed to send Telegram alert: {e}")
            finally:
                if os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except Exception:
                        pass

        threading.Thread(target=_do_send, name=f"TgAlert-{cam_name}", daemon=True).start()


class CameraWorker:
    """Worker handling motion gating and triggering inference for a single camera."""

    def __init__(
        self,
        config: CameraConfig,
        detector: Optional[SharedYOLODetector],
        notifier: TelegramNotifier,
        enable_ai: bool = True,
    ):
        self.config = config
        self.detector = detector
        self.notifier = notifier
        self.enable_ai = enable_ai
        self.stream = RTSPFFmpegStream(
            name=config.name,
            url=config.url,
            transport=config.transport,
            width=STREAM_WIDTH,
            height=STREAM_HEIGHT,
            fps=STREAM_FPS,
            threads=FF_THREADS,
        )
        self.motion_gate = MotionGate()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.last_ai_time = 0.0
        self.motion_events = 0
        self.human_events = 0

    def start(self):
        self._running = True
        self.stream.start()
        self._thread = threading.Thread(target=self._loop, name=f"Worker-{self.config.name}", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.stream.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _loop(self):
        logger.info(f"[{self.config.name}] Worker loop started.")
        last_log_time = time.time()

        while self._running:
            has_frame, frame = self.stream.get_frame()
            if not has_frame or frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            is_motion, delta_pixels = self.motion_gate.check_motion(frame)

            # Periodic status log every 15 seconds if idle
            if (now - last_log_time) >= 15.0:
                last_log_time = now
                logger.info(f"[{self.config.name}] Status: idle (last delta={delta_pixels}px, thresh={MOTION_THRESHOLD})")

            if is_motion:
                self.motion_events += 1
                logger.info(f"[{self.config.name}] MOTION DETECTED (delta={delta_pixels}px >= {MOTION_THRESHOLD})")

                # Throttle AI calls
                if self.enable_ai and self.detector and (now - self.last_ai_time) >= AI_MIN_INTERVAL:
                    self.last_ai_time = now
                    has_human, highest_conf, boxes = self.detector.detect_human(frame)

                    if has_human:
                        self.human_events += 1
                        logger.info(f"[{self.config.name}] HUMAN CONFIRMED! (conf={highest_conf:.2f}, boxes={len(boxes)})")
                        self.notifier.send_alert(self.config.name, highest_conf, frame, boxes)
                    else:
                        logger.info(f"[{self.config.name}] AI Checked: Motion is not a human (filtered).")


def main():
    parser = argparse.ArgumentParser(description="Lightweight Multi-Stream CCTV Human Detection")
    parser.add_argument("--no-ai", action="store_true", help="Run motion gating only without AI inference")
    parser.add_argument("--camera", type=str, choices=["kamera1", "kamera2", "all"], default="all", help="Camera filter")
    parser.add_argument("--test-snapshot", action="store_true", help="Capture 1 snapshot from each camera and exit")
    parser.add_argument("--test-telegram", action="store_true", help="Send a test notification to Telegram and exit")
    parser.add_argument("--duration", type=int, default=0, help="Run for N seconds then exit (0 = infinite)")
    args = parser.parse_args()

    all_cameras = get_configured_cameras()
    if not all_cameras:
        logger.error("No cameras configured in .env! Please configure CAM1_* or CAM2_*.")
        sys.exit(1)

    selected_cameras = all_cameras
    if args.camera != "all":
        selected_cameras = [c for c in all_cameras if c.name == args.camera]

    logger.info(f"Selected cameras: {[c.name for c in selected_cameras]}")

    # Mode: Test Snapshot
    if args.test_snapshot:
        logger.info("Running snapshot test...")
        for c in selected_cameras:
            out_file = f"/tmp/snapshot_{c.name}.jpg"
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-rtsp_transport", c.transport,
                "-i", c.url,
                "-vf", f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=fast_bilinear",
                "-vframes", "1",
                out_file,
            ]
            rc = subprocess.call(cmd)
            if rc == 0 and os.path.exists(out_file):
                logger.info(f"[{c.name}] Snapshot success -> {out_file} ({os.path.getsize(out_file)} bytes)")
            else:
                logger.error(f"[{c.name}] Snapshot failed!")
        return

    # Mode: Test Telegram Alert
    if args.test_telegram:
        logger.info("Sending test Telegram alert with snapshot...")
        notifier = TelegramNotifier()
        if not notifier.active:
            logger.error("Telegram is not active. Check TELEGRAM_TOKEN and TELEGRAM_CHAT_ID.")
            return
        c = selected_cameras[0]
        test_frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
        # Try grab a real snapshot
        out_file = f"/tmp/snapshot_{c.name}.jpg"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", c.transport,
            "-i", c.url,
            "-vf", f"scale={STREAM_WIDTH}:{STREAM_HEIGHT}:flags=fast_bilinear",
            "-vframes", "1",
            out_file,
        ]
        rc = subprocess.call(cmd)
        if rc == 0 and os.path.exists(out_file):
            loaded = cv2.imread(out_file)
            if loaded is not None:
                test_frame = loaded
        fake_boxes = [([50, 50, 100, 150], 0.95)]
        notifier.send_alert(c.name, 0.95, test_frame, fake_boxes)
        time.sleep(3)  # wait for async thread
        logger.info("Test Telegram alert process triggered.")
        return

    # Initialize AI Detector
    detector = None
    if not args.no_ai:
        try:
            detector = SharedYOLODetector(MODEL_PATH, PROB_THRESHOLD)
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")
            sys.exit(1)
    else:
        logger.info("Running in NO-AI mode (Motion Gate evaluation only).")

    # Initialize Alerter
    notifier = TelegramNotifier()
    if notifier.active:
        logger.info(f"Telegram alerting active for chat ID {notifier.chat_id}.")
    else:
        logger.warning("Telegram alerting inactive (token or chat_id empty).")

    # Spawn Camera Workers
    workers = []
    for c in selected_cameras:
        w = CameraWorker(c, detector, notifier, enable_ai=(not args.no_ai))
        workers.append(w)
        w.start()

    logger.info("All camera workers active. Press Ctrl+C to terminate.")

    # Handle graceful exit
    stop_event = threading.Event()

    def _sig_handler(sig, frame):
        logger.info("Termination signal received. Shutting down gracefully...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    start_time = time.time()
    try:
        while not stop_event.is_set():
            time.sleep(1)
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                logger.info(f"Duration limit of {args.duration}s reached. Stopping...")
                break
    finally:
        for w in workers:
            w.stop()
        logger.info("Detector stopped cleanly.")


if __name__ == "__main__":
    main()
