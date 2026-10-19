"""
RTSP FFmpeg stream reader with pipe watchdog and automatic reconnection.
"""

import time
import select
import logging
import subprocess
import threading
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("CCTV-Detector.Stream")


class RTSPFFmpegStream:
    """Manages an FFmpeg child process reading RTSP and piping raw BGR frames."""

    def __init__(
        self,
        name: str,
        url: str,
        transport: str = "tcp",
        width: int = 320,
        height: int = 320,
        fps: int = 1,
        threads: int = 1,
        read_timeout: float = 6.0,
    ):
        self.name = name
        self.url = url
        self.transport = transport
        self.width = width
        self.height = height
        self.fps = fps
        self.threads = threads
        self.read_timeout = read_timeout
        self.frame_size = width * height * 3

        self.proc: Optional[subprocess.Popen] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._has_new_frame = False

    def start(self) -> None:
        """Starts the background stream reader thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name=f"RTSP-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Terminates FFmpeg process and stops reader thread."""
        self._running = False
        self._terminate_proc()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _terminate_proc(self) -> None:
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

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

    def _read_exact(self, num_bytes: int) -> Optional[bytes]:
        """Reads exactly num_bytes from stdout pipe with select-based I/O timeout."""
        buf = bytearray()
        last_read_time = time.time()
        while len(buf) < num_bytes and self._running and self.proc and self.proc.poll() is None:
            rlist, _, _ = select.select([self.proc.stdout], [], [], 1.0)
            if not rlist:
                if (time.time() - last_read_time) >= self.read_timeout:
                    logger.warning(f"[{self.name}] Stream data timeout ({self.read_timeout}s without data). Reconnecting...")
                    return None
                continue

            needed = num_bytes - len(buf)
            chunk = self.proc.stdout.read(min(needed, 65536))
            if not chunk:
                return None
            buf.extend(chunk)
            last_read_time = time.time()

        return bytes(buf) if len(buf) == num_bytes else None

    def _run_loop(self) -> None:
        logger.info(f"[{self.name}] Stream reader started ({self.transport.upper()}, {self.fps} FPS, {self.width}x{self.height})")
        while self._running:
            self.proc = self._start_ffmpeg()
            if not self.proc:
                time.sleep(3.0)
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

            self._terminate_proc()

            if self._running:
                logger.warning(f"[{self.name}] Stream lost. Reconnecting in 3s...")
                time.sleep(3.0)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Atomically retrieves the latest frame if available."""
        with self._lock:
            if self._has_new_frame and self._latest_frame is not None:
                self._has_new_frame = False
                return True, self._latest_frame.copy()
            return False, None
