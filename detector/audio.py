"""
Audio alert playback using ALSA (aplay) with cooldown management.
"""

import os
import time
import logging
import subprocess
import threading

logger = logging.getLogger("CCTV-Detector.Audio")


class AudioAlerter:
    """Plays an audio sound effect asynchronously when a detection event occurs."""

    def __init__(
        self,
        sound_file: str = "assets/ding.wav",
        cooldown_seconds: int = 15,
        enabled: bool = True,
    ):
        self.sound_file = sound_file
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled
        self.last_play_time = 0.0
        self._lock = threading.Lock()

    def play(self) -> None:
        """Plays the configured sound file if enabled and cooldown has elapsed."""
        if not self.enabled:
            return

        if not self.sound_file or not os.path.exists(self.sound_file):
            logger.warning(f"Audio file not found: {self.sound_file}")
            return

        now = time.time()
        with self._lock:
            if (now - self.last_play_time) < self.cooldown_seconds:
                return
            self.last_play_time = now

        def _do_play():
            try:
                subprocess.run(
                    ["aplay", "-q", self.sound_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                    check=False,
                )
                logger.info(f"Audio chime played: {os.path.basename(self.sound_file)}")
            except Exception as e:
                logger.warning(f"Failed to play audio alert: {e}")

        threading.Thread(target=_do_play, name="AudioAlertPlayer", daemon=True).start()
