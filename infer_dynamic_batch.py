import os
import glob
import time
import random
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

IMAGE_DIR = "../COCO-SMALL-3/train/images"
INPUT_SIZE = 640
INPUT_NAME = "images"
OUTPUT_NAME = "output0"

def preprocess(image, size=640):
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw = int(w * scale)
    nh = int(h * scale)
    resized = cv2.resize(image, (nw, nh))
    pad_w = size - nw
    pad_h = size - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    norm = rgb.astype(np.float32) / 255.0
    chw = np.transpose(norm, (2, 0, 1))
    return chw  # (3,640,640)

class TRTModel:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = trt_runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        print("[INFO] Engine loaded.")
        print("[INFO] Engine input:", self.engine.get_tensor_shape(INPUT_NAME))
        print("[INFO] Engine output:", self.engine.get_tensor_shape(OUTPUT_NAME))

    def infer_batch(self, batch):
        """
        batch: numpy array (B,3,640,640)
        """
        batch = np.ascontiguousarray(batch, dtype=np.float32)  # 👈 FIX CONTIGUOUS
        B = batch.shape[0]

        # Dynamic shape
        self.context.set_input_shape(INPUT_NAME, batch.shape)

        # Lấy output shape thực
        out_shape = self.context.get_tensor_shape(OUTPUT_NAME)
        out_shape = tuple(out_shape)

        d_in = cuda.mem_alloc(batch.nbytes)
        output = np.empty(out_shape, dtype=np.float32)
        d_out = cuda.mem_alloc(output.nbytes)

        # Set bindings
        self.context.set_tensor_address(INPUT_NAME, int(d_in))
        self.context.set_tensor_address(OUTPUT_NAME, int(d_out))

        # Copy input
        cuda.memcpy_htod_async(d_in, batch, self.stream)

        t0 = time.time()
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(output, d_out, self.stream)
        self.stream.synchronize()
        t1 = time.time()

        d_in.free()
        d_out.free()

        return output, (t1 - t0) * 1000

def load_images(n=40):
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if len(images) == 0:
        raise RuntimeError("No images found")

    chosen = random.sample(images, min(n, len(images)))
    return chosen

def collate_batch(image_paths):
    batch = []
    for p in image_paths:
        img = cv2.imread(p)
        pre = preprocess(img)
        batch.append(pre)
    batch = np.stack(batch, axis=0)  # (B,3,640,640)
    return batch

def main(engine_path):
    model = TRTModel(engine_path)

    imgs = load_images(40)
    print(f"[INFO] Found {len(imgs)} images")

    for B in [1, 2, 4, 8]:
        print("\n==============================================")
        print(f"[TEST] Batch size = {B}")
        print("==============================================")

        batch_paths = random.sample(imgs, B)
        batch = collate_batch(batch_paths)

        out, t = model.infer_batch(batch)
        print(f"[RESULT] Output shape: {out.shape}, time = {t:.2f} ms")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, required=True)
    args = parser.parse_args()
    main(args.engine)
