# build_engine.py
import os
from pathlib import Path
import glob

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

# ============================================================
# Config
# ============================================================
ONNX_FILE = "weights/yolov8n.onnx"

FP16_ENGINE = "weights/yolov8n_fp16.engine"
INT8_ENGINE = "weights/yolov8n_int8.engine"

CALIB_IMAGE_DIR = "COCO-SMALL-3/train/images"
CALIB_IMAGE_LIMIT = 200  # Giới hạn số ảnh calibration (tăng để chính xác hơn, giảm để nhanh hơn)

INPUT_C = 3
INPUT_H = 640
INPUT_W = 640
INPUT_TENSOR_NAME = "images"  # tên input trong ONNX

WORKSPACE_SIZE = 1 << 30  # 1 GB

# ============================================================
# Preprocess dùng cho calibration (letterbox + normalize)
# ============================================================
def preprocess_calib(image, h=INPUT_H, w=INPUT_W):
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    new_w = int(iw * scale)
    new_h = int(ih * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = w - new_w
    pad_h = h - new_h
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    norm = rgb.astype(np.float32) / 255.0
    chw = np.transpose(norm, (2, 0, 1))
    batch = np.expand_dims(chw, axis=0)  # (1,3,640,640)
    return np.ascontiguousarray(batch, dtype=np.float32)

# ============================================================
# Calibrator INT8
# ============================================================
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths, cache_file="calib_cache.bin"):
        super().__init__()
        self.image_paths = image_paths
        self.cache_file = cache_file
        self.index = 0

        self.batch_size = 1  # quan trọng: N=1
        self.input_nbytes = int(
            self.batch_size * INPUT_C * INPUT_H * INPUT_W * np.float32().nbytes
        )
        self.device_input = cuda.mem_alloc(self.input_nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.image_paths):
            return None

        img_path = self.image_paths[self.index]
        img = cv2.imread(img_path)
        if img is None:
            data = np.zeros((1, INPUT_C, INPUT_H, INPUT_W), dtype=np.float32)
            print(f"[CALIB] Failed to read {img_path}, using zeros.")
        else:
            data = preprocess_calib(img)

        cuda.memcpy_htod(self.device_input, data)
        self.index += 1
        print(f"[CALIB] Providing batch {self.index}/{len(self.image_paths)}: {img_path}")
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print("[CALIB] Using existing calibration cache.")
            with open(self.cache_file, "rb") as f:
                return f.read()
        print("[CALIB] No calibration cache found, will run calibration.")
        return None

    def write_calibration_cache(self, cache):
        print("[CALIB] Writing calibration cache.")
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# ============================================================
# Helper tạo builder + network
# ============================================================
def create_builder_and_network():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(ONNX_FILE, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX:")
            for i in range(parser.num_errors):
                print("   ", parser.get_error(i))
            return None, None, None
    return builder, network, parser

# ============================================================
# Thêm optimization profile cho dynamic batch
# ============================================================
def add_dynamic_batch_profile(builder, config, network):
    input_tensor = network.get_input(0)
    name = input_tensor.name
    print(f"[INFO] Network input name: {name}, shape: {input_tensor.shape}")

    # Nếu tên khác "images" thì vẫn dùng tên thực tế
    if name != INPUT_TENSOR_NAME:
        print(f"[WARN] INPUT_TENSOR_NAME='{INPUT_TENSOR_NAME}' "
              f"khác với '{name}', dùng '{name}' theo ONNX.")
    # Profile: N dynamic, H/W cố định 640, nhưng opt = 1 để khớp calibrator
    profile = builder.create_optimization_profile()
    min_shape = (1, INPUT_C, INPUT_H, INPUT_W)
    opt_shape = (1, INPUT_C, INPUT_H, INPUT_W)   # !!! quan trọng
    max_shape = (8, INPUT_C, INPUT_H, INPUT_W)

    profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    print(f"[INFO] Added optimization profile for '{name}': "
          f"min={min_shape}, opt={opt_shape}, max={max_shape}")

# ============================================================
# Main build
# ============================================================
def build_engines(build_fp16=True, build_int8=True):
    # ---- Chuẩn bị danh sách ảnh calib ----
    # Try multiple locations for calibration directory
    candidates = [
        Path(CALIB_IMAGE_DIR),  # relative to CWD
        Path(__file__).resolve().parents[1] / CALIB_IMAGE_DIR,  # repo root
        Path(__file__).resolve().parent / CALIB_IMAGE_DIR,  # script dir
    ]
    
    calib_dir = None
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            calib_dir = str(candidate)
            print(f"[INFO] Using calibration directory: {calib_dir}")
            break
    
    if calib_dir is None:
        tried = [str(c) for c in candidates]
        raise RuntimeError(
            f"Calibration directory not found: {CALIB_IMAGE_DIR}\n"
            f"Tried: {', '.join(tried)}"
        )

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    calib_images = []
    for pat in patterns:
        calib_images.extend(glob.glob(os.path.join(calib_dir, pat)))
    calib_images = sorted(calib_images)
    
    # Limit số ảnh calibration
    if len(calib_images) > CALIB_IMAGE_LIMIT:
        print(f"[INFO] Limiting calibration images from {len(calib_images)} to {CALIB_IMAGE_LIMIT}")
        calib_images = calib_images[:CALIB_IMAGE_LIMIT]
    
    if not calib_images:
        raise RuntimeError(f"No calibration images found in {calib_dir}")

    print(f"[INFO] Found {len(calib_images)} calibration images")

    # ---- FP16 engine ----
    if build_fp16:
        builder, network, parser = create_builder_and_network()
        if builder is None:
            return False
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_SIZE)
        add_dynamic_batch_profile(builder, config, network)
        config.set_flag(trt.BuilderFlag.FP16)

        print("[INFO] Building FP16...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("[ERROR] FP16 engine build failed.")
            return False
        with open(FP16_ENGINE, "wb") as f:
            f.write(engine_bytes)
        print(f"[INFO] Engine saved: {FP16_ENGINE}")

    # ---- INT8 engine ----
    if build_int8:
        builder, network, parser = create_builder_and_network()
        if builder is None:
            return False
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_SIZE)
        add_dynamic_batch_profile(builder, config, network)

        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = EntropyCalibrator(calib_images)
        config.int8_calibrator = calibrator

        print("[INFO] Building INT8 PTQ...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("[ERROR] INT8 engine build failed.")
            return False
        with open(INT8_ENGINE, "wb") as f:
            f.write(engine_bytes)
        print(f"[INFO] Engine saved: {INT8_ENGINE}")

    return True

if __name__ == "__main__":
    Path("weights").mkdir(exist_ok=True)
    ok = build_engines(build_fp16=True, build_int8=True)
    print("[INFO] Build", "succeeded." if ok else "failed.")
