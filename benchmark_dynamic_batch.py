#!/usr/bin/env python3
"""
Dynamic Batch Inference với TensorRT engines
Test nhiều batch sizes khác nhau: 1, 2, 4, 8
"""
import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import time
from collections import defaultdict

# ============================================================================
# Config
# ============================================================================
FP16_ENGINE = "weights/yolov8n_fp16.engine"
INT8_ENGINE = "weights/yolov8n_int8.engine"

IMAGE_DIR = "../COCO-SMALL-3/train/images"
BATCH_SIZES = [1, 2, 4, 8]  # Test các batch size khác nhau
NUM_IMAGES_PER_BATCH = 10  # Số ảnh để test mỗi batch size

INPUT_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.5

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# ============================================================================
# Preprocess
# ============================================================================
def preprocess_image(image, input_size=INPUT_SIZE):
    """Preprocess một ảnh thành (3, 640, 640)"""
    h, w = image.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_w = input_size - new_w
    pad_h = input_size - new_h
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    
    return chw, scale, pad_top, pad_left


def preprocess_batch(images, input_size=INPUT_SIZE):
    """Preprocess batch ảnh thành (N, 3, 640, 640)"""
    batch_data = []
    metadata = []
    
    for img in images:
        chw, scale, pad_h, pad_w = preprocess_image(img, input_size)
        batch_data.append(chw)
        metadata.append((img.shape[0], img.shape[1], scale, pad_h, pad_w))
    
    batch_array = np.stack(batch_data, axis=0).astype(np.float32)
    return np.ascontiguousarray(batch_array), metadata


# ============================================================================
# Postprocess
# ============================================================================
def postprocess_single(output, orig_h, orig_w, scale, pad_h, pad_w):
    """Postprocess output của 1 ảnh trong batch"""
    # YOLOv8 output: (84, 8400) -> transpose -> (8400, 84)
    output = output.T
    
    scores = output[:, 4:]
    best_scores = np.max(scores, axis=1)
    best_classes = np.argmax(scores, axis=1)
    
    mask = best_scores >= CONF_THRESH
    boxes = output[mask, :4]
    scores = best_scores[mask]
    classes = best_classes[mask]
    
    if len(scores) == 0:
        return []
    
    # Convert xywh -> xyxy và scale back
    xc, yc, w, h = boxes.T
    xc = (xc - pad_w) / scale
    yc = (yc - pad_h) / scale
    w /= scale
    h /= scale
    
    x1 = np.clip(xc - w/2, 0, orig_w - 1)
    y1 = np.clip(yc - h/2, 0, orig_h - 1)
    x2 = np.clip(xc + w/2, 0, orig_w - 1)
    y2 = np.clip(yc + h/2, 0, orig_h - 1)
    
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes_xyxy, scores, IOU_THRESH)
    
    return [
        (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
         float(scores[i]), int(classes[i]))
        for i in keep
    ]


def nms(boxes, scores, iou_thresh):
    """Non-maximum suppression"""
    if len(boxes) == 0:
        return []
    
    idxs = np.argsort(scores)[::-1]
    keep = []
    
    while len(idxs) > 0:
        cur = idxs[0]
        keep.append(cur)
        if len(idxs) == 1:
            break
        
        ious = calc_iou(boxes[cur], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thresh]
    
    return keep


def calc_iou(box, boxes):
    """Calculate IoU"""
    x1, y1, x2, y2 = box
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])
    
    inter_w = np.clip(xx2 - xx1, 0, None)
    inter_h = np.clip(yy2 - yy1, 0, None)
    inter = inter_w * inter_h
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    return inter / (area1 + area2 - inter + 1e-6)


# ============================================================================
# TensorRT Dynamic Batch Inference
# ============================================================================
class TensorRTDynamicBatch:
    def __init__(self, engine_path):
        # Load engine
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        self.input_name = "images"
        self.output_name = "output0"
        
        self.stream = cuda.Stream()
        
        print(f"[INFO] Loaded engine: {engine_path}")
        print(f"  Input shape: {self.engine.get_tensor_shape(self.input_name)}")
        print(f"  Output shape: {self.engine.get_tensor_shape(self.output_name)}")
    
    def infer_batch(self, images):
        """
        Inference với batch ảnh
        images: list of numpy arrays (OpenCV images)
        Returns: list of detections (one per image), inference time (ms)
        """
        batch_size = len(images)
        
        # Preprocess batch
        batch_data, metadata = preprocess_batch(images)
        
        # Set dynamic shape
        self.context.set_input_shape(self.input_name, batch_data.shape)
        
        # Get output shape
        output_shape = self.context.get_tensor_shape(self.output_name)
        
        # Allocate memory
        d_input = cuda.mem_alloc(batch_data.nbytes)
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)
        
        # Copy input to device
        cuda.memcpy_htod_async(d_input, batch_data, self.stream)
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(d_input))
        self.context.set_tensor_address(self.output_name, int(d_output))
        
        # Execute
        start = time.time()
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.stream.synchronize()
        infer_time = (time.time() - start) * 1000
        
        # Free memory
        d_input.free()
        d_output.free()
        
        # Postprocess each image in batch
        all_detections = []
        for i in range(batch_size):
            orig_h, orig_w, scale, pad_h, pad_w = metadata[i]
            dets = postprocess_single(output[i], orig_h, orig_w, scale, pad_h, pad_w)
            all_detections.append(dets)
        
        return all_detections, infer_time


# ============================================================================
# Load test images
# ============================================================================
def load_test_images(image_dir, num_images):
    """Load random images để test"""
    image_paths = sorted(Path(image_dir).glob("*.jpg"))
    
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {image_dir}")
    
    # Random sample
    import random
    selected = random.sample(image_paths, min(num_images, len(image_paths)))
    
    images = []
    for p in selected:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
    
    return images


# ============================================================================
# Benchmark
# ============================================================================
def benchmark_dynamic_batch(engine_path, engine_name):
    """Test engine với các batch sizes khác nhau"""
    print("\n" + "="*80)
    print(f"BENCHMARKING: {engine_name}")
    print("="*80)
    
    detector = TensorRTDynamicBatch(engine_path)
    
    results = defaultdict(list)
    
    for batch_size in BATCH_SIZES:
        print(f"\n[BATCH SIZE = {batch_size}]")
        
        # Load images cho batch này
        images = load_test_images(IMAGE_DIR, batch_size * NUM_IMAGES_PER_BATCH)
        
        # Chạy NUM_IMAGES_PER_BATCH lần
        for run_idx in range(NUM_IMAGES_PER_BATCH):
            batch_images = images[run_idx * batch_size : (run_idx + 1) * batch_size]
            
            if len(batch_images) < batch_size:
                break
            
            detections, infer_time = detector.infer_batch(batch_images)
            
            total_dets = sum(len(d) for d in detections)
            throughput = batch_size / (infer_time / 1000)  # images/sec
            
            results[batch_size].append({
                'time': infer_time,
                'detections': total_dets,
                'throughput': throughput
            })
            
            if run_idx < 3:  # Print first 3 runs
                print(f"  Run {run_idx+1}: {infer_time:.2f} ms, "
                      f"{total_dets} detections, "
                      f"{throughput:.1f} img/s")
    
    # Summary
    print(f"\n[SUMMARY - {engine_name}]")
    print(f"{'Batch':>6s} {'Avg Time (ms)':>15s} {'Avg Throughput':>15s} {'Per-image (ms)':>15s}")
    print("-" * 60)
    
    for batch_size in BATCH_SIZES:
        if batch_size not in results or len(results[batch_size]) == 0:
            continue
        
        times = [r['time'] for r in results[batch_size]]
        throughputs = [r['throughput'] for r in results[batch_size]]
        
        avg_time = np.mean(times)
        avg_throughput = np.mean(throughputs)
        per_image_time = avg_time / batch_size
        
        print(f"{batch_size:>6d} {avg_time:>15.2f} {avg_throughput:>15.1f} {per_image_time:>15.2f}")


# ============================================================================
# Main
# ============================================================================
def main():
    """
    Usage:
      python benchmark_dynamic_batch.py fp16    -> test FP16 engine
      python benchmark_dynamic_batch.py int8    -> test INT8 engine
      python benchmark_dynamic_batch.py all     -> test cả 2
      python benchmark_dynamic_batch.py         -> mặc định: all
    """
    import sys
    
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if mode in ["fp16", "all"]:
        if not os.path.exists(FP16_ENGINE):
            print(f"[ERROR] FP16 engine not found: {FP16_ENGINE}")
        else:
            benchmark_dynamic_batch(FP16_ENGINE, "FP16")
    
    if mode in ["int8", "all"]:
        if not os.path.exists(INT8_ENGINE):
            print(f"[ERROR] INT8 engine not found: {INT8_ENGINE}")
        else:
            benchmark_dynamic_batch(INT8_ENGINE, "INT8")
    
    print("\n[INFO] Benchmark completed!")


if __name__ == "__main__":
    main()
