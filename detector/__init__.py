"""
CCTV Human Detection Module.
Modular architecture adhering to Single Responsibility and Clean Code principles.
"""

from .config import AppConfig, CameraConfig, load_config
from .stream import RTSPFFmpegStream
from .motion import MotionGate
from .inference import SharedYOLODetector
from .notifier import TelegramNotifier
from .audio import AudioAlerter
from .worker import CameraWorker

__all__ = [
    "AppConfig",
    "CameraConfig",
    "load_config",
    "RTSPFFmpegStream",
    "MotionGate",
    "SharedYOLODetector",
    "TelegramNotifier",
    "AudioAlerter",
    "CameraWorker",
]
