import os
import cv2
import numpy as np
import requests
import time
import onnxruntime as ort
import threading
import logging
import subprocess
import shlex
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_env_file(env_path=".env"):
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        logging.warning(f"Gagal memuat file .env: {exc}")


# === KONFIGURASI (dari environment) ===
load_env_file()

# Menggunakan SRT via MediaMTX (Low Latency, High Quality)
STREAM_URL = os.getenv('STREAM_URL', 'srt://127.0.0.1:8890?streamid=read:kamera1')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/yolov8n.onnx')
STREAM_WIDTH = int(os.getenv('STREAM_WIDTH', '640'))
STREAM_HEIGHT = int(os.getenv('STREAM_HEIGHT', '640'))
STREAM_FPS = int(os.getenv('STREAM_FPS', '8'))
FF_THREADS = int(os.getenv('FF_THREADS', '1'))            # limit ffmpeg threads
JPEG_QUALITY = int(os.getenv('JPEG_QUALITY', '8'))       # 2-31 lower -> higher quality; higher number = lower CPU
AI_INTERVAL = float(os.getenv('AI_INTERVAL', '5'))       # minimum seconds between AI inferences
USE_HWACCEL = str(os.getenv('USE_HWACCEL', 'false')).lower() in ('1', 'true', 'yes')

# Telegram config
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
_chat_ids_env = os.getenv('TELEGRAM_CHAT_IDS', '')
if _chat_ids_env:
    TELEGRAM_CHAT_IDS = [x.strip() for x in _chat_ids_env.split(',') if x.strip()]
elif TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_IDS = [TELEGRAM_CHAT_ID]
else:
    TELEGRAM_CHAT_IDS = []

MOTION_THRESHOLD = int(os.getenv('MOTION_THRESHOLD', '500'))
PROB_THRESHOLD = float(os.getenv('PROB_THRESHOLD', '0.5'))
MOTION_DELTA_THRESHOLD = int(os.getenv('MOTION_DELTA_THRESHOLD', '18'))
MOTION_BLUR_SIZE = int(os.getenv('MOTION_BLUR_SIZE', '15'))
MOTION_DILATE_ITERATIONS = int(os.getenv('MOTION_DILATE_ITERATIONS', '1'))
MOTION_ALPHA = float(os.getenv('MOTION_ALPHA', '0.3'))

# === SRT STREAM READER ===
class SRTFFmpegReader:
    """Read SRT stream by running ffmpeg and piping raw frames to stdout.

    This avoids relying on OpenCV's build having SRT support.
    It forces a fixed frame size for easier parsing.
    """
    def __init__(self, stream_url, width=640, height=480, fps=10, ff_threads=1, jpeg_quality=8, use_hwaccel=False):
        self.stream_url = stream_url
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.ff_threads = int(ff_threads)
        self.jpeg_quality = int(jpeg_quality)
        self.use_hwaccel = use_hwaccel
        self.frame_size = self.width * self.height * 3
        self.proc = None
        self.frame = None
        self.status = False
        self._stop = False
        self.thread = threading.Thread(target=self._reader_thread, args=())
        self.thread.daemon = True
        self.thread.start()

    def _start_process(self):
        # Use MJPEG image2pipe to avoid fixed-size raw frame issues and make parsing robust
        # Increase probe values so codecs (HEVC) and stream size can be detected reliably
        # Limit ffmpeg threads and scale/fps to reduce CPU on low-end devices.
        hwaccel_flags = ""
        if self.use_hwaccel:
            # Intel VAAPI hardware acceleration for H.264/HEVC decoding
            hwaccel_flags = "-hwaccel vaapi -hwaccel_device /dev/dri/renderD128 "
        
        cmd = (
            f'ffmpeg -hide_banner -loglevel warning -fflags nobuffer -flags low_delay '
            f'-probesize 5000000 -analyzeduration 1000000 {hwaccel_flags}-threads {self.ff_threads} -i "{self.stream_url}" '
            f'-vf scale={self.width}:{self.height},fps={self.fps} -f image2pipe -vcodec mjpeg -q:v {self.jpeg_quality} pipe:1'
        )
        args = shlex.split(cmd)
        try:
            self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            logging.warning(f"Gagal menjalankan ffmpeg: {e}")
            self.proc = None

    def _reader_thread(self):
        while not self._stop:
            if self.proc is None:
                self._start_process()
                if self.proc is None:
                    time.sleep(1)
                    continue

            try:
                # Read jpeg frames from stdout by finding JPEG SOI/EOI markers
                buf = b''
                while not self._stop:
                    chunk = self.proc.stdout.read(4096)
                    if not chunk:
                        # process ended or no data
                        break
                    buf += chunk
                    # look for JPEG frame boundaries
                    start = buf.find(b'\xff\xd8')
                    end = buf.find(b'\xff\xd9', start + 2)
                    if start != -1 and end != -1:
                        jpg = buf[start:end+2]
                        buf = buf[end+2:]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is None:
                            continue
                        # ensure size matches requested (cv2.resize if needed)
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))
                        self.frame = frame
                        self.status = True
                        # yield one frame then continue reading
                        break
                # if we exited inner loop due to no chunk, restart ffmpeg
                if not buf and (self.proc is None or self.proc.poll() is not None):
                    self.status = False
                    self.frame = None
                    if self.proc:
                        try:
                            stderr = self.proc.stderr.read().decode('utf-8', errors='ignore')
                            logging.debug(f'ffmpeg stderr: {stderr}')
                        except Exception:
                            pass
                    if self.proc:
                        try:
                            self.proc.kill()
                        except Exception:
                            pass
                    self.proc = None
                    time.sleep(0.1)
                    continue
            except Exception:
                self.status = False
                self.frame = None
                if self.proc:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                self.proc = None
                time.sleep(0.2)

    def read(self):
        return self.status, self.frame

    def stop(self):
        self._stop = True
        if self.proc:
            try:
                self.proc.kill()
            except Exception:
                pass

# === INISIALISASI AI ===
def ensure_model_available(model_path=MODEL_PATH):
    """Ensure the ONNX model exists locally. If missing, attempt to run the helper script to fetch/export it.

    This will call `scripts/get_yolo_models.py` with the current Python interpreter.
    """
    if os.path.exists(model_path):
        return True

    logging.info(f"Model tidak ditemukan di {model_path}. Mencoba mengunduh/mengekspor menggunakan helper script...")
    script = os.path.join(os.path.dirname(__file__), 'scripts', 'get_yolo_models.py')
    if not os.path.exists(script):
        logging.error(f"Helper script tidak ditemukan: {script}. Silakan buat atau jalankan ekspor secara manual.")
        return False

    try:
        rc = subprocess.call([sys.executable, script])
        if rc != 0:
            logging.error(f"Helper script mengembalikan kode keluar {rc}.")
            return False
    except Exception as e:
        logging.exception(f"Gagal menjalankan helper script: {e}")
        return False

    if os.path.exists(model_path):
        logging.info(f"Model tersedia di {model_path} setelah proses helper.")
        return True
    logging.error(f"Model masih tidak ditemukan setelah menjalankan helper: {model_path}")
    return False

print("Memuat Model YOLOv8 ONNX...")
# ONNX runtime session options tuned for low-end device (limit threads)
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1
try:
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
except Exception:
    pass

# Ensure model exists; if not, attempt to download/export it.
if not ensure_model_available(MODEL_PATH):
    logging.error("Model ONNX tidak tersedia. Hentikan program.")
    raise SystemExit(1)

session = ort.InferenceSession(MODEL_PATH, sess_options=sess_opts, providers=['CPUExecutionProvider'])
outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]

# Instantiate ffmpeg SRT reader with low-end settings
stream = SRTFFmpegReader(STREAM_URL, width=STREAM_WIDTH, height=STREAM_HEIGHT, fps=STREAM_FPS, ff_threads=FF_THREADS, jpeg_quality=JPEG_QUALITY, use_hwaccel=USE_HWACCEL)
print("Sistem Pintar SRT Teroptimasi Dimulai...")

# --- Telegram helper (send message or photo asynchronously, fallback to logging)
def send_telegram_or_log(text, photo_path=None):
    def _send():
        if TELEGRAM_TOKEN and TELEGRAM_TOKEN != "TOKEN_BOT_MU" and str(TELEGRAM_TOKEN).lower() != "null":
            # Resolve recipients: support TELEGRAM_CHAT_IDS (list) or TELEGRAM_CHAT_ID (single)
            recipients = []
            try:
                if isinstance(TELEGRAM_CHAT_IDS, (list, tuple)) and len(TELEGRAM_CHAT_IDS) > 0:
                    recipients = [str(x) for x in TELEGRAM_CHAT_IDS]
                elif isinstance(TELEGRAM_CHAT_IDS, str) and TELEGRAM_CHAT_IDS.strip():
                    recipients = [x.strip() for x in TELEGRAM_CHAT_IDS.split(',') if x.strip()]
            except NameError:
                pass
            # fallback to TELEGRAM_CHAT_ID
            if not recipients and TELEGRAM_CHAT_ID:
                recipients = [str(TELEGRAM_CHAT_ID)]

            for rcpt in recipients:
                try:
                    if photo_path:
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                            data={'chat_id': rcpt, 'caption': text},
                            files={'photo': open(photo_path, 'rb')},
                            timeout=10
                        )
                    else:
                        requests.post(
                            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                            data={'chat_id': rcpt, 'text': text},
                            timeout=10
                        )
                except Exception as e:
                    logging.warning(f"Gagal kirim Telegram ke {rcpt}: {e}")
        else:
            logging.info(text)
    threading.Thread(target=_send, daemon=True).start()

def detect_human(frame):
    # Preprocessing (YOLOv8 butuh 640x640)
    h, w = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 640))
    image_data = np.array(image_resized) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    
    # Inference
    outputs = session.run(outname, {inname[0]: image_data})
    output = outputs[0][0].transpose()
    
    boxes = []
    confidences = []
    
    x_factor = w / 640.0
    y_factor = h / 640.0
    
    for box in output:
        prob = box[4:]
        class_id = np.argmax(prob)
        conf = prob[class_id]
        if class_id == 0 and conf > PROB_THRESHOLD:
            cx, cy, bw, bh = box[0], box[1], box[2], box[3]
            x = int((cx - bw / 2) * x_factor)
            y = int((cy - bh / 2) * y_factor)
            width = int(bw * x_factor)
            height = int(bh * y_factor)
            boxes.append([x, y, width, height])
            confidences.append(float(conf))
            
    is_human = False
    highest_conf = 0.0
    final_boxes = []
    
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, PROB_THRESHOLD, 0.4)
        if len(indices) > 0:
            is_human = True
            for i in indices.flatten():
                final_boxes.append((boxes[i], confidences[i]))
                if confidences[i] > highest_conf:
                    highest_conf = confidences[i]

    return is_human, highest_conf, final_boxes

# === LOOP UTAMA ===
avg_frame = None
last_ai_time = 0

current_boxes = []
box_expiration = 0

# Tampilan placeholder jika stream terputus / belum connect
blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
last_reconnect_log = 0

running = True
prev_status = False
while running:
    status, frame = stream.read()
    # Connection state change notifications
    if status and not prev_status:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        send_telegram_or_log(f"✅ Terhubung ke stream SRT\nWaktu: {ts}\nURL: {STREAM_URL}")
    if (not status) and prev_status:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        send_telegram_or_log(f"❌ Terputus dari stream SRT\nWaktu: {ts}\nURL: {STREAM_URL}")
    prev_status = status
    if not status or frame is None:
        frame_display = blank_frame.copy()
        cv2.putText(frame_display, "Menghubungkan / Stream Offline...", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("CCTV Stream - Human Detection", frame_display)
        
        if (time.time() - last_reconnect_log) > 3:
            logging.warning("Tidak dapat membaca frame dari stream SRT. Menunggu koneksi...")
            last_reconnect_log = time.time()
            
        if cv2.waitKey(30) & 0xFF == ord('q'):
            running = False
            break
            
        time.sleep(0.1)
        continue
        
    frame_display = frame.copy()

    # 1. Deteksi Gerakan (Sangat Ringan)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_blur = MOTION_BLUR_SIZE if MOTION_BLUR_SIZE % 2 == 1 else MOTION_BLUR_SIZE + 1
    gray = cv2.GaussianBlur(gray, (motion_blur, motion_blur), 0)

    if avg_frame is None:
        avg_frame = gray.astype("float")
        continue

    cv2.accumulateWeighted(gray, avg_frame, MOTION_ALPHA)
    frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg_frame))
    thresh = cv2.threshold(frame_delta, MOTION_DELTA_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=MOTION_DILATE_ITERATIONS)
    
    motion_count = np.sum(thresh) / 255
    
    # 2. Eksekusi AI jika ada gerakan & lolos batas waktu 5 detik
    if motion_count > MOTION_THRESHOLD and (time.time() - last_ai_time) > AI_INTERVAL:
        last_ai_time = time.time()
        print(f"Gerakan Terdeteksi! Memproses SRT Frame lewat AI...")
        
        is_human, confidence, detected_boxes = detect_human(frame)
        if is_human:
            current_boxes = detected_boxes
            box_expiration = time.time() + AI_INTERVAL  # Tampilkan bounding box selama AI_INTERVAL detik
            
            # Gambar bounding box juga di gambar yang mau disimpan/dikirim
            for (box, conf) in current_boxes:
                x, y, bw, bh = box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                cv2.putText(frame, f"Human {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
            foto_temp = "/tmp/srt_detected.jpg"
            cv2.imwrite(foto_temp, frame) # Kualitas asli bawaan stream CCTV
            
            # Kirim Telegram atau log (asynchronous) agar tidak memblokir loop utama
            timestamp = time.strftime("%H:%M:%S")
            caption = f"⚠️ MANUSIA TERDETEKSI (SRT)\nJam: {timestamp}\nAkurasi: {confidence:.2f}"
            send_telegram_or_log(caption, photo_path=foto_temp)

    # ===== MENAMPILKAN GUI DENGAN BOUNDING BOX =====
    if time.time() < box_expiration:
        for (box, conf) in current_boxes:
            x, y, bw, bh = box
            cv2.rectangle(frame_display, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.putText(frame_display, f"Human {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    cv2.imshow("CCTV Stream - Human Detection", frame_display)
    
    # Render GUI OpenCV, refresh key per 30ms (~30 FPS max rendering)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        running = False
        break

# Graceful shutdown: send notification, stop stream, close windows
try:
    send_telegram_or_log(f"⏹️ Sistem AI CCTV dimatikan. Waktu: {time.strftime('%Y-%m-%d %H:%M:%S')}")
except Exception:
    pass
stream.stop()
cv2.destroyAllWindows()
time.sleep(0.2)
