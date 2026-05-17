#!/usr/bin/env python3
"""Download and export YOLOv8n models into ./models/.

This script attempts to use the `ultralytics` package to obtain the pretrained
`yolov8n` model and export it to ONNX. It requires `ultralytics` to be
installed in the active Python environment.

Usage:
  source cctv_env/bin/activate
  pip install ultralytics
  python3 scripts/get_yolo_models.py

The script will create `models/` (if missing) and place `yolov8n.onnx` there.
If you already have `models/yolov8n.pt`, the script will use it to export ONNX.
"""
import os
import sys
import subprocess
import glob
import shutil


def ensure_models_dir():
    os.makedirs('models', exist_ok=True)


def run_export_with_ultralytics():
    # Run a short Python one-liner that loads the official yolov8n weights and exports ONNX
    cmd = [sys.executable, '-c', "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"]
    print('Running:', ' '.join(cmd))
    return subprocess.call(cmd)


def find_and_move_onnx(dest='models/yolov8n.onnx'):
    # Find any recently created .onnx file (common exporters write to CWD)
    onnx_files = glob.glob('*.onnx')
    if not onnx_files:
        return False
    # choose most recently modified
    onnx_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    src = onnx_files[0]
    print('Found ONNX:', src, '->', dest)
    shutil.move(src, dest)
    return True


def main():
    ensure_models_dir()

    # If models/yolov8n.pt exists, export from it; otherwise rely on ultralytics to fetch weights
    local_pt = os.path.join('models', 'yolov8n.pt')
    if os.path.exists(local_pt):
        print('Using local PT:', local_pt)
        cmd = [sys.executable, '-c', f"from ultralytics import YOLO; YOLO('{local_pt}').export(format='onnx')"]
        rc = subprocess.call(cmd)
    else:
        print('No local PT found; attempting to fetch and export using ultralytics (requires internet)')
        rc = run_export_with_ultralytics()

    if rc != 0:
        print('\nERROR: ultralytics export failed. Make sure `pip install ultralytics` and try again.')
        sys.exit(1)

    if not find_and_move_onnx():
        print('\nERROR: export finished but no .onnx file found in the current directory.')
        sys.exit(2)

    print('\nDone. ONNX model saved to models/yolov8n.onnx')


if __name__ == '__main__':
    main()
