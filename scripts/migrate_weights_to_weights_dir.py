#!/usr/bin/env python3
"""
Migration helper: move existing model/checkpoint files into `weights/` directory.

Run from the `ONNX-and-TensorRT-Infer` folder:
    python scripts/migrate_weights_to_weights_dir.py

It will move files if they exist and avoid overwriting existing files in `weights/`.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os_files = [
    ROOT / 'ONNX' / 'yolov8n.pt',
    ROOT / 'ONNX' / 'yolov8n.onnx',
    ROOT / 'ONNX' / 'yolov8n.engine',
    ROOT / 'ONNX' / 'yolov8n_fp16.engine',
    ROOT / 'ONNX' / 'yolov8n_int8.engine',
    ROOT / 'ONNX' / 'yolov8n_qdq.onnx',
    ROOT / 'yolov8n.pt',
    ROOT / 'yolov8n.onnx',
    ROOT / 'yolov8n.engine',
]

DEST = ROOT / 'weights'
DEST.mkdir(exist_ok=True)

def move_file(src: Path, dst_dir: Path):
    if not src.exists():
        return False
    dst = dst_dir / src.name
    if dst.exists():
        print(f"[SKIP] Destination exists: {dst}")
        return False
    print(f"[MOVE] {src} -> {dst}")
    shutil.move(str(src), str(dst))
    return True

def main():
    moved = 0
    for p in os_files:
        if move_file(p, DEST):
            moved += 1
    print(f"Done. Files moved: {moved}")

if __name__ == '__main__':
    main()
