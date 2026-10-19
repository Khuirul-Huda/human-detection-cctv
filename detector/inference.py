"""
Thread-safe YOLOv8 ONNX inference engine.
"""

import os
import time
import logging
import threading
from typing import List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("CCTV-Detector.AI")


class SharedYOLODetector:
    """Thread-safe single-session YOLOv8 ONNX inference engine."""

    def __init__(self, model_path: str = "models/yolov8n.onnx", prob_threshold: float = 0.5):
        self.model_path = model_path
        self.prob_threshold = prob_threshold
        self._lock = threading.Lock()
        self.session: Optional[ort.InferenceSession] = None
        self.input_name = ""
        self.output_name = ""
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        logger.info(f"Loading YOLO model: {self.model_path} (Single Core CPU)...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info("YOLO ONNX model loaded successfully.")

    def detect_human(self, frame: np.ndarray) -> Tuple[bool, float, List[Tuple[List[int], float]]]:
        """
        Detects class 0 (person) objects in the provided frame.
        Returns:
            (has_person, highest_conf, [(box, conf)]) where box is [x, y, w, h].
        """
        h, w = frame.shape[:2]
        # Preprocessing: convert BGR to RGB and resize to 640x640
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        img_data = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
        img_data = np.expand_dims(img_data, axis=0)

        # Thread-safe serialized inference
        with self._lock:
            t0 = time.time()
            outputs = self.session.run([self.output_name], {self.input_name: img_data})
            inference_ms = (time.time() - t0) * 1000
            logger.debug(f"Inference completed in {inference_ms:.1f}ms")

        output = outputs[0][0].transpose()  # shape: [8400, 84]

        boxes: List[List[int]] = []
        confidences: List[float] = []
        x_factor = w / 640.0
        y_factor = h / 640.0

        for row in output:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])

            # class_id 0 = person in COCO dataset
            if class_id == 0 and conf >= self.prob_threshold:
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                x = int((cx - bw / 2) * x_factor)
                y = int((cy - bh / 2) * y_factor)
                width = int(bw * x_factor)
                height = int(bh * y_factor)
                boxes.append([x, y, width, height])
                confidences.append(conf)

        if not boxes:
            return False, 0.0, []

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.prob_threshold, 0.45)
        final_boxes: List[Tuple[List[int], float]] = []
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
