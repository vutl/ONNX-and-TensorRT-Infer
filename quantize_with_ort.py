import os
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

# ============================================================
# Config
# ============================================================
FP32_ONNX_PATH = "ONNX/yolov8n.onnx"
INT8_QDQ_ONNX_PATH = "ONNX/yolov8n_qdq.onnx"

# Ảnh dùng để calibration (khuyến nghị tăng số lượng)
CALIB_IMAGE_PATHS = [
    "ONNX/bus.jpg",
    "ONNX/car.jpg",
    "ONNX/person.jpg",
    # có thể thêm nhiều ảnh khác
]

INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, 640, 640)

# ============================================================
def preprocess_for_onnx(image, input_shape):
    h, w = image.shape[:2]
    target_size = input_shape[2]

    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    input_data = np.expand_dims(chw, axis=0)
    input_data = np.ascontiguousarray(input_data, dtype=np.float32)
    return input_data

# ============================================================
class YoloDataReader(CalibrationDataReader):
    def __init__(self, image_paths, input_name, input_shape):
        self.image_paths = image_paths
        self.input_name = input_name
        self.input_shape = input_shape
        self._data_iter = None

    def get_next(self):
        if self._data_iter is None:
            input_list = []
            for img_path in self.image_paths:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARN] Failed to read image: {img_path}, using zeros.")
                    inp = np.zeros(self.input_shape, dtype=np.float32)
                else:
                    inp = preprocess_for_onnx(img, self.input_shape)
                input_list.append({self.input_name: inp})
            self._data_iter = iter(input_list)

        try:
            return next(self._data_iter)
        except StopIteration:
            return None

# ============================================================
def main():
    if not os.path.exists(FP32_ONNX_PATH):
        raise FileNotFoundError(f"Model not found: {FP32_ONNX_PATH}")

    model = onnx.load(FP32_ONNX_PATH)
    graph = model.graph
    if len(graph.input) > 0:
        detected_input_name = graph.input[0].name
        if detected_input_name != INPUT_NAME:
            print(f"[INFO] Detected input name in ONNX: {detected_input_name}")
            print(f"[INFO] Config INPUT_NAME = '{INPUT_NAME}'. Please adjust if needed.")

    print(f"[INFO] Using input name for quantization: {INPUT_NAME}")

    data_reader = YoloDataReader(CALIB_IMAGE_PATHS, INPUT_NAME, INPUT_SHAPE)

    os.makedirs(os.path.dirname(INT8_QDQ_ONNX_PATH), exist_ok=True)

    print(f"[QDQ] Quantizing model (static, QDQ) from:")
    print(f"      {FP32_ONNX_PATH}")
    print(f"      → {INT8_QDQ_ONNX_PATH}")

    quantize_static(
        model_input=FP32_ONNX_PATH,
        model_output=INT8_QDQ_ONNX_PATH,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )

    print("[QDQ] Done. Saved quantized QDQ ONNX model.")

if __name__ == "__main__":
    main()
