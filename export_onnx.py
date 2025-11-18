from ultralytics import YOLO
import os

# Config
PT_MODEL_PATH = "ONNX/yolov8n.pt"
ONNX_OUTPUT_PATH = "ONNX/yolov8n.onnx"
IMG_SIZE = 640
OPSET_VERSION = 12

def export_to_onnx():
    if not os.path.exists(PT_MODEL_PATH):
        raise FileNotFoundError(f"Model .pt not found: {PT_MODEL_PATH}")

    model = YOLO(PT_MODEL_PATH)
    print(f"[INFO] Exporting model {PT_MODEL_PATH} → {ONNX_OUTPUT_PATH}")

    model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=OPSET_VERSION,
        simplify=True,
        dynamic=False
    )

    print(f"[INFO] ONNX model saved: {ONNX_OUTPUT_PATH}")

if __name__ == "__main__":
    export_to_onnx()
