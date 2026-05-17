# Human Detection CCTV

SRT-based CCTV human detection with YOLOv8 ONNX, OpenCV GUI, Telegram alerts, and environment-based configuration.

## Project Layout

- `main.py` - application entry point
- `.env` - local runtime configuration, not committed
- `.env.example` - template for local setup
- `models/` - ignored folder for model weights
- `cctv_env/` - local Python virtual environment

## Setup

1. Copy the example env file:

```bash
cp .env.example .env
```

2. Create the virtual environment if it does not exist, then activate it:

```bash
python3 -m venv cctv_env
source cctv_env/bin/activate
```

3. Install dependencies from the pinned list:

```bash
pip install -r requirements.txt
```

4. Edit `.env` and set your values:

- `STREAM_URL` - your SRT input URL
- `MODEL_PATH` - default is `models/yolov8n.onnx`
- `TELEGRAM_TOKEN` - bot token from BotFather
- `TELEGRAM_CHAT_ID` or `TELEGRAM_CHAT_IDS` - one or more recipients

5. Make sure the model files exist in `models/`:

- `models/yolov8n.onnx`
- `models/yolov8n.pt`

If you want to replace them, keep the same path or update `MODEL_PATH` in `.env`.

### Where to get the model files

- `models/yolov8n.pt` can be downloaded from the official Ultralytics YOLOv8 release or model package, then placed into `models/`.
- `models/yolov8n.onnx` can be exported from `models/yolov8n.pt` with Ultralytics after you install the package locally.

Example export command:

```bash
python -c "from ultralytics import YOLO; YOLO('models/yolov8n.pt').export(format='onnx')"
```

After export, move the generated `.onnx` file into `models/yolov8n.onnx` if the exporter wrote it elsewhere.

### Automatic helper script

There is a helper script that attempts to fetch the official `yolov8n` weights via `ultralytics` and export ONNX into `models/`:

```bash
source cctv_env/bin/activate
pip install ultralytics
python3 scripts/get_yolo_models.py
```

The script will place `models/yolov8n.onnx` into the `models/` folder. If you already have a local `models/yolov8n.pt`, the script will use it.

## Run

```bash
source cctv_env/bin/activate
python3 main.py
```

If you use a different virtual environment path, activate it first before running the app. The `cctv_env/` folder is gitignored on purpose, so each user recreates it locally.

Press `q` in the GUI window to exit cleanly.

## Notes

- Model weights are intentionally kept out of git.
- `.gitignore` already excludes `models/` and `weights/`.
- Motion sensitivity can be tuned from `.env` using:
  - `MOTION_THRESHOLD`
  - `MOTION_DELTA_THRESHOLD`
  - `MOTION_BLUR_SIZE`
  - `MOTION_DILATE_ITERATIONS`
  - `MOTION_ALPHA`
