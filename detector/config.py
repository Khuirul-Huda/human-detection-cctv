"""
Configuration management for CCTV Human Detection.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("CCTV-Detector.Config")


@dataclass
class CameraConfig:
    name: str
    url: str
    transport: str = "tcp"
    enabled: bool = True
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class AppConfig:
    # Stream settings
    stream_fps: int = 1
    stream_width: int = 320
    stream_height: int = 320
    ff_threads: int = 1

    # AI Model settings
    model_path: str = "models/yolov8n.onnx"
    prob_threshold: float = 0.5
    ai_min_interval: float = 3.0

    # Motion gate settings
    motion_threshold: int = 400
    motion_delta_threshold: int = 18
    motion_blur_size: int = 15
    motion_dilate_iterations: int = 1
    motion_alpha: float = 0.3
    motion_warmup_frames: int = 8

    # Telegram Alert settings
    telegram_token: str = ""
    telegram_chat_id: str = ""
    alert_cooldown_seconds: int = 60

    # Audio Alert settings
    enable_audio_alert: bool = True
    audio_alert_file: str = "assets/ding.wav"
    audio_cooldown_seconds: int = 15

    # Camera list
    cameras: List[CameraConfig] = field(default_factory=list)


def _find_env_file(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds .env file from custom path, current working directory, or project root."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    if os.path.exists(".env"):
        return os.path.abspath(".env")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_env = os.path.join(project_root, ".env")
    if os.path.exists(root_env):
        return root_env
    return None


def _load_env_file(env_path: Optional[str] = None) -> None:
    """Populates os.environ from .env if variables are not already set."""
    resolved_path = _find_env_file(env_path)
    if not resolved_path or not os.path.exists(resolved_path):
        return
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
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
        logger.warning(f"Failed to load environment file {resolved_path}: {e}")


def load_config(env_path: Optional[str] = None) -> AppConfig:
    """Loads application configuration from environment variables and .env file."""
    _load_env_file(env_path)

    cameras: List[CameraConfig] = []
    # Discover cameras CAM1, CAM2, ...
    for i in (1, 2, 3, 4):
        prefix = f"CAM{i}_"
        enabled = str(os.getenv(f"{prefix}ENABLE", "false")).lower() in ("1", "true", "yes")
        name = os.getenv(f"{prefix}NAME", f"kamera{i}")
        url = os.getenv(f"{prefix}URL", "")
        transport = os.getenv(f"{prefix}TRANSPORT", "tcp")

        # Resolve native sub-stream resolution
        w_env = os.getenv(f"{prefix}WIDTH")
        h_env = os.getenv(f"{prefix}HEIGHT")
        default_w = 480 if name == "kamera1" else 640
        default_h = 544 if name == "kamera1" else 360
        cam_width = int(w_env) if w_env else default_w
        cam_height = int(h_env) if h_env else default_h

        if enabled and url:
            cameras.append(
                CameraConfig(
                    name=name,
                    url=url,
                    transport=transport,
                    enabled=True,
                    width=cam_width,
                    height=cam_height,
                )
            )

    # Backward compatibility fallback
    if not cameras and os.getenv("STREAM_URL"):
        cameras.append(
            CameraConfig(
                name="kamera1",
                url=os.getenv("STREAM_URL", ""),
                transport="tcp",
                enabled=True,
                width=480,
                height=544,
            )
        )

    model_path = os.getenv("MODEL_PATH", "models/yolov8n.onnx")
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(project_root, model_path)
        if os.path.exists(candidate):
            model_path = candidate

    audio_file = os.getenv("AUDIO_ALERT_FILE", "assets/ding.wav")
    if not os.path.isabs(audio_file) and not os.path.exists(audio_file):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate_audio = os.path.join(project_root, audio_file)
        if os.path.exists(candidate_audio):
            audio_file = candidate_audio

    enable_audio = str(os.getenv("ENABLE_AUDIO_ALERT", "true")).lower() in ("1", "true", "yes")

    return AppConfig(
        stream_fps=int(os.getenv("STREAM_FPS", "1")),
        stream_width=int(os.getenv("STREAM_WIDTH", "320")),
        stream_height=int(os.getenv("STREAM_HEIGHT", "320")),
        ff_threads=int(os.getenv("FF_THREADS", "1")),
        model_path=model_path,
        prob_threshold=float(os.getenv("PROB_THRESHOLD", "0.5")),
        ai_min_interval=float(os.getenv("AI_MIN_INTERVAL", "3.0")),
        motion_threshold=int(os.getenv("MOTION_THRESHOLD", "400")),
        motion_delta_threshold=int(os.getenv("MOTION_DELTA_THRESHOLD", "18")),
        motion_blur_size=int(os.getenv("MOTION_BLUR_SIZE", "15")),
        motion_dilate_iterations=int(os.getenv("MOTION_DILATE_ITERATIONS", "1")),
        motion_alpha=float(os.getenv("MOTION_ALPHA", "0.3")),
        motion_warmup_frames=int(os.getenv("MOTION_WARMUP_FRAMES", "8")),
        telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        alert_cooldown_seconds=int(os.getenv("ALERT_COOLDOWN_SECONDS", "60")),
        enable_audio_alert=enable_audio,
        audio_alert_file=audio_file,
        audio_cooldown_seconds=int(os.getenv("AUDIO_COOLDOWN_SECONDS", "15")),
        cameras=cameras,
    )
