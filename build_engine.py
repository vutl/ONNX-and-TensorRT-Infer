import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import cv2

# ============================================================================
ONNX_FILE = "ONNX/yolov8n.onnx"
FP16_ENGINE = "ONNX/yolov8n_fp16.engine"
INT8_ENGINE = "ONNX/yolov8n_int8.engine"
CALIB_IMAGES = [
    "ONNX/bus.jpg",
    "ONNX/car.jpg",
    "ONNX/person.jpg",
    # thêm nhiều ảnh nếu có
]
INPUT_SHAPE = (1, 3, 640, 640)
WORKSPACE_SIZE = 1 << 30
INPUT_TENSOR_NAME = "images"

# ============================================================================
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths, input_shape, cache_file="calib_cache.bin"):
        super(Calibrator, self).__init__()
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.index = 0
        self.input_nbytes = int(np.prod(self.input_shape) * np.float32().nbytes)
        self.device_input = cuda.mem_alloc(self.input_nbytes)

    def get_batch_size(self):
        return self.input_shape[0]

    def get_batch(self, names):
        if self.index >= len(self.image_paths):
            return None
        img_path = self.image_paths[self.index]
        img = cv2.imread(img_path)
        if img is None:
            batch = np.zeros(self.input_shape, dtype=np.float32)
        else:
            h, w = img.shape[:2]
            target = self.input_shape[2]
            scale = min(target/w, target/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img, (new_w,new_h))
            pad_w = target - new_w
            pad_h = target - new_h
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            padded = cv2.copyMakeBorder(
                resized,
                pad_top, pad_h-pad_top,
                pad_left, pad_w-pad_left,
                cv2.BORDER_CONSTANT, value=(114,114,114)
            )
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            norm = rgb.astype(np.float32)/255.0
            chw = np.transpose(norm, (2,0,1))
            batch = np.expand_dims(chw, axis=0).astype(np.float32)
        cuda.memcpy_htod(self.device_input, batch)
        print(f"[CALIB] Providing batch {self.index+1}/{len(self.image_paths)}: {img_path}")
        self.index += 1
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

# ============================================================================
def build_engine(fp16=True, int8=False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(ONNX_FILE, 'rb') as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX file!")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_SIZE)

    # optimization profile (fix static if bạn muốn)
    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_TENSOR_NAME,
                      min=INPUT_SHAPE,
                      opt=INPUT_SHAPE,
                      max=INPUT_SHAPE)
    config.add_optimization_profile(profile)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] Building FP16 engine...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("[ERROR] FP16 engine build failed.")
            return False
        with open(FP16_ENGINE, "wb") as f:
            f.write(engine_bytes)
        print(f"[INFO] FP16 engine saved to: {FP16_ENGINE}")

    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = Calibrator(CALIB_IMAGES, INPUT_SHAPE)
        config.int8_calibrator = calibrator
        print("[INFO] Building INT8 engine (calibration)...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            print("[ERROR] INT8 engine build failed.")
            return False
        with open(INT8_ENGINE, "wb") as f:
            f.write(engine_bytes)
        print(f"[INFO] INT8 engine saved to: {INT8_ENGINE}")

    return True

if __name__ == "__main__":
    Path(os.path.dirname(FP16_ENGINE)).mkdir(parents=True, exist_ok=True)
    build_engine(fp16=True, int8=True)
