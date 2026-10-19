"""
Telegram notification handler with cooldown and annotated image delivery.
"""

import os
import time
import logging
import threading
from typing import Dict, List, Tuple
import cv2
import numpy as np
import requests

logger = logging.getLogger("CCTV-Detector.Notifier")


class TelegramNotifier:
    """Delivers detection alert photos with bounding boxes to Telegram."""

    def __init__(self, token: str, chat_id: str, cooldown_seconds: int = 60):
        self.token = token
        self.chat_id = chat_id
        self.cooldown = cooldown_seconds
        self.last_alert_time: Dict[str, float] = {}
        self.active = bool(token and token != "TOKEN_BOT_MU" and chat_id)

    def can_alert(self, cam_name: str) -> bool:
        """Checks if alert cooldown for the specified camera has elapsed."""
        now = time.time()
        last = self.last_alert_time.get(cam_name, 0.0)
        return (now - last) >= self.cooldown

    def send_alert(
        self,
        cam_name: str,
        highest_conf: float,
        frame: np.ndarray,
        boxes: List[Tuple[List[int], float]],
    ) -> None:
        """Annotates frame and asynchronously transmits photo alert to Telegram."""
        if not self.active:
            logger.info(f"[{cam_name}] ALERT: Human detected ({highest_conf * 100:.0f}%), Telegram not configured.")
            return

        if not self.can_alert(cam_name):
            logger.info(f"[{cam_name}] Human detected ({highest_conf * 100:.0f}%), alert skipped due to cooldown.")
            return

        self.last_alert_time[cam_name] = time.time()

        # Render bounding boxes onto a copy of the frame
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
                        timeout=12.0,
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
