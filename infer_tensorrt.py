import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import sys
import time


# ============================================================================
# Configuration
# ============================================================================
ENGINE_PATH = "ONNX/yolov8n.engine"
IMAGE_PATHS = [
    "ONNX/bus.jpg",
    "ONNX/car.jpg",
    "ONNX/person.jpg"
]
OUTPUT_DIR = "output"
INPUT_SIZE = 640  # YOLOv8n input size
CONF_THRESH = 0.5
IOU_THRESH = 0.45


# COCO classes
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


# Color palette for drawing boxes
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


# ============================================================================
# Helper Function: Preprocess (Static)
# ============================================================================
def preprocess_static(image, input_shape):
    """
    Static preprocessing function dùng cho cả inference và calibration
    
    Args:
        image: numpy array (H, W, 3) in BGR format
        input_shape: tuple (1, 3, 640, 640)
        
    Returns:
        input_data: (1, 3, 640, 640) in float32
        scale_h, scale_w: scale factors
        pad_h, pad_w: padding offsets
    """
    h, w = image.shape[:2]
    target_size = input_shape[2]
    
    # Calculate scale to fit image into target_size while maintaining aspect ratio
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create letterbox (pad to 640x640)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Convert BGR -> RGB
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    
    # Normalize [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    # HWC -> CHW
    chw = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension: CHW -> NCHW
    input_data = np.expand_dims(chw, axis=0)
    input_data = np.ascontiguousarray(input_data)
    
    return input_data, scale, scale, pad_top, pad_left


# ============================================================================
# TensorRT Engine Loader
# ============================================================================
class YOLOv8TensorRT:
    def __init__(self, engine_path):
        """Load TensorRT engine"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_tensor_shape("images")
        self.input_dtype = self.engine.get_tensor_dtype("images")
        self.output_shape = self.engine.get_tensor_shape("output0")
        
        print(f"[INFO] Input shape: {self.input_shape}")
        print(f"[INFO] Output shape: {self.output_shape}")
        
        # Tạo 1 stream dùng chung
        self.stream = cuda.Stream()

    def infer(self, image):
        """
        Run inference on single image
        
        Args:
            image: numpy array (H, W, 3) in BGR format
            
        Returns:
            detections: list of (x1, y1, x2, y2, conf, class_id)
            inference_time: time in milliseconds
        """
        # Preprocessing
        input_data, scale_h, scale_w, pad_h, pad_w = preprocess_static(image, self.input_shape)
        
        # Allocate GPU buffers
        d_input = cuda.mem_alloc(input_data.nbytes)
        output = np.empty(self.output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)
        
        # Copy input to GPU
        cuda.memcpy_htod_async(d_input, input_data, self.stream)
        
        # Set tensor addresses
        self.context.set_tensor_address("images", int(d_input))
        self.context.set_tensor_address("output0", int(d_output))
        
        # Run inference with timing
        start_time = time.time()
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy output from GPU to host
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.stream.synchronize()
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Free GPU memory
        d_input.free()
        d_output.free()
        
        # Postprocess
        detections = self.postprocess(
            output, image.shape[0], image.shape[1], scale_h, scale_w, pad_h, pad_w
        )
        
        return detections, inference_time

    def postprocess(self, output, orig_h, orig_w, scale_h, scale_w, pad_h, pad_w):
        """
        Postprocess YOLOv8 output
        
        YOLOv8 output format: (1, 84, 8400)
        - First 4 values: [x_center, y_center, width, height] (in model coordinates)
        - Next 80 values: class confidences
        
        Returns:
            detections: list of (x1, y1, x2, y2, conf, class_id)
        """
        # output shape: (1, 84, 8400)
        output = output[0]  # (84, 8400)
        
        # Transpose to (8400, 84) for easier processing
        output = output.T  # (8400, 84)
        
        # Get max class confidence for each prediction
        class_scores = output[:, 4:]  # (8400, 80)
        max_scores = np.max(class_scores, axis=1)  # (8400,)
        class_ids = np.argmax(class_scores, axis=1)  # (8400,)
        
        # Filter by confidence threshold
        mask = max_scores >= CONF_THRESH
        
        # Extract valid predictions
        valid_boxes = output[mask, :4]  # (N, 4) - xywh format
        valid_scores = max_scores[mask]  # (N,)
        valid_classes = class_ids[mask]  # (N,)
        
        if len(valid_scores) == 0:
            return []
        
        # Convert from model coordinates to original image coordinates
        x_center = valid_boxes[:, 0]
        y_center = valid_boxes[:, 1]
        width = valid_boxes[:, 2]
        height = valid_boxes[:, 3]
        
        # Remove padding and scale
        x_center = (x_center - pad_w) / scale_h
        y_center = (y_center - pad_h) / scale_w
        width = width / scale_h
        height = height / scale_w
        
        # Convert xywh to xyxy (x1, y1, x2, y2)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)
        
        # Apply NMS
        boxes = np.column_stack([x1, y1, x2, y2])
        keep_idx = self.nms(boxes, valid_scores, IOU_THRESH)
        
        detections = []
        for idx in keep_idx:
            detections.append((
                float(x1[idx]),
                float(y1[idx]),
                float(x2[idx]),
                float(y2[idx]),
                float(valid_scores[idx]),
                int(valid_classes[idx])
            ))
        
        return detections

    @staticmethod
    def nms(boxes, scores, iou_threshold):
        """
        Non-Maximum Suppression
        
        Args:
            boxes: (N, 4) in xyxy format
            scores: (N,)
            iou_threshold: float
            
        Returns:
            keep_idx: indices of boxes to keep
        """
        if len(boxes) == 0:
            return []
        
        # Sort by score (highest first)
        sorted_idx = np.argsort(scores)[::-1]
        
        keep_idx = []
        while len(sorted_idx) > 0:
            current_idx = sorted_idx[0]
            keep_idx.append(current_idx)
            
            if len(sorted_idx) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current_idx]
            remaining_boxes = boxes[sorted_idx[1:]]
            ious = YOLOv8TensorRT.calc_iou(current_box, remaining_boxes)
            
            # Keep only boxes with IoU below threshold
            keep_mask = ious < iou_threshold
            sorted_idx = sorted_idx[1:][keep_mask]
        
        return keep_idx

    @staticmethod
    def calc_iou(box, boxes):
        """Calculate IoU between one box and multiple boxes"""
        x1_min, y1_min, x1_max, y1_max = box
        
        x2_min = boxes[:, 0]
        y2_min = boxes[:, 1]
        x2_max = boxes[:, 2]
        y2_max = boxes[:, 3]
        
        # Intersection
        inter_x_min = np.maximum(x1_min, x2_min)
        inter_y_min = np.maximum(y1_min, y2_min)
        inter_x_max = np.minimum(x1_max, x2_max)
        inter_y_max = np.minimum(y1_max, y2_max)
        
        inter_w = np.clip(inter_x_max - inter_x_min, 0, None)
        inter_h = np.clip(inter_y_max - inter_y_min, 0, None)
        inter_area = inter_w * inter_h
        
        # Union
        box_area = (x1_max - x1_min) * (y1_max - y1_min)
        boxes_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box_area + boxes_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou


# ============================================================================
# CALIBRATOR FOR INT8
# ============================================================================
class SimpleCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Calibrator for INT8 quantization
    Dùng ảnh thực tế để hiệu chuẩn mô hình
    """
    def __init__(self, image_paths, input_shape):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.input_shape = input_shape  # (1, 3, 640, 640)
        self.index = 0
        self.image_paths = image_paths
        self.cache_file = "calib_cache.bin"

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        """Trả về một batch dữ liệu calibration"""
        if self.index >= len(self.image_paths):
            return None
        
        # Load ảnh
        image = cv2.imread(self.image_paths[self.index])
        if image is None:
            # Nếu file lỗi, trả về batch zeros
            batch_img = np.zeros(self.input_shape, dtype=np.float32)
        else:
            # Preprocess ảnh
            batch_img, _, _, _, _ = preprocess_static(image, self.input_shape)
        
        self.index += 1
        
        # Trả về pointer mảng numpy
        return [batch_img.ctypes.data]

    def read_calibration_cache(self):
        """Đọc cache calibration nếu có"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Lưu cache calibration"""
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ============================================================================
# Build Engine Functions
# ============================================================================
def build_tensorrt_engine_fp16(onnx_file, engine_file):
    """Build TensorRT engine with FP16 precision"""
    print(f"\n[INFO] Parsing ONNX file: {onnx_file}")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_file, 'rb') as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX file!")
            for err_num in range(parser.num_errors):
                print(f"   {parser.get_error(err_num)}")
            return False
    
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    print("[INFO] Building FP16 TensorRT engine...")
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        print("[ERROR] Failed to build FP16 engine!")
        return False
    
    with open(engine_file, "wb") as f:
        f.write(engine)
    
    print(f"[INFO] FP16 engine saved to: {engine_file}")
    return True


def build_tensorrt_engine_int8(onnx_file, engine_file, calib_image_paths, input_shape):
    """Build TensorRT engine with INT8 precision using calibration"""
    print(f"\n[INFO] Parsing ONNX file: {onnx_file}")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_file, 'rb') as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX file!")
            for err_num in range(parser.num_errors):
                print(f"   {parser.get_error(err_num)}")
            return False
    
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    
    # Setup calibrator
    calibrator = SimpleCalibrator(calib_image_paths, input_shape)
    config.int8_calibrator = calibrator
    
    print("[INFO] Building INT8 TensorRT engine (this may take a while due to calibration)...")
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        print("[ERROR] Failed to build INT8 engine!")
        return False
    
    with open(engine_file, "wb") as f:
        f.write(engine)
    
    print(f"[INFO] INT8 engine saved to: {engine_file}")
    return True


# ============================================================================
# Visualization
# ============================================================================
def draw_boxes(image, detections, title="Detection"):
    """Draw bounding boxes on image"""
    result = image.copy()
    
    for x1, y1, x2, y2, conf, class_id in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw box
        color = tuple(map(int, COLORS[class_id]))
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{COCO_CLASSES[class_id]}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        label_y = y1 - 5 if y1 - text_size[1] - 10 > 0 else y2 + text_size[1] + 10
        
        cv2.rectangle(
            result,
            (x1, label_y - text_size[1] - 5),
            (x1 + text_size[0], label_y + 5),
            color,
            -1
        )
        cv2.putText(result, label, (x1, label_y), font, font_scale, (255, 255, 255), thickness)
    
    # Add title
    cv2.putText(result, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return result


def draw_comparison(image, fp16_detections, int8_detections):
    """Draw side-by-side comparison of FP16 and INT8 detections"""
    h, w = image.shape[:2]
    
    # Draw FP16 on left
    fp16_result = draw_boxes(image.copy(), fp16_detections, f"FP16 ({len(fp16_detections)} detections)")
    
    # Draw INT8 on right
    int8_result = draw_boxes(image.copy(), int8_detections, f"INT8 ({len(int8_detections)} detections)")
    
    # Concatenate horizontally
    comparison = np.hstack([fp16_result, int8_result])
    
    return comparison


# ============================================================================
# Inference Function
# ============================================================================
def run_inference(detector, images, engine_name="Engine"):
    """Run inference on a list of images"""
    for img_path in images:
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue
        
        print(f"\n[INFO] Processing: {img_path}")
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Failed to read image: {img_path}")
            continue
        
        h, w = image.shape[:2]
        print(f"  Image size: {w}x{h}")
        
        # Run inference
        detections, infer_time = detector.infer(image)
        print(f"  Detections: {len(detections)}")
        print(f"  Inference time: {infer_time:.2f} ms")
        
        for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
            print(f"    [{i}] {COCO_CLASSES[class_id]}: conf={conf:.3f}, box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        # Draw boxes
        result = draw_boxes(image, detections, f"{engine_name} ({len(detections)} detections)")
        
        # Save output
        output_path = os.path.join(OUTPUT_DIR, f"result_{engine_name}_{Path(img_path).stem}.jpg")
        cv2.imwrite(output_path, result)
        print(f"  Saved to: {output_path}")


def compare_engines(fp16_detector, int8_detector, images):
    """Run inference on both FP16 and INT8 engines and create comparison images"""
    print("\n" + "="*80)
    print("COMPARING FP16 vs INT8 ENGINES")
    print("="*80)
    
    for img_path in images:
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue
        
        print(f"\n[INFO] Processing: {img_path}")
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Failed to read image: {img_path}")
            continue
        
        h, w = image.shape[:2]
        print(f"  Image size: {w}x{h}")
        
        # Run FP16 inference
        print("  Running FP16 inference...")
        fp16_detections, fp16_time = fp16_detector.infer(image)
        print(f"    FP16 detections: {len(fp16_detections)}, Time: {fp16_time:.2f} ms")
        for i, (x1, y1, x2, y2, conf, class_id) in enumerate(fp16_detections):
            print(f"      [{i}] {COCO_CLASSES[class_id]}: {conf:.3f}")
        
        # Run INT8 inference
        print("  Running INT8 inference...")
        int8_detections, int8_time = int8_detector.infer(image)
        print(f"    INT8 detections: {len(int8_detections)}, Time: {int8_time:.2f} ms")
        for i, (x1, y1, x2, y2, conf, class_id) in enumerate(int8_detections):
            print(f"      [{i}] {COCO_CLASSES[class_id]}: {conf:.3f}")
        
        # Print comparison stats
        print(f"\n  [COMPARISON]")
        print(f"    Detection difference: {abs(len(fp16_detections) - len(int8_detections))} boxes")
        print(f"    Speed improvement (INT8 vs FP16): {(fp16_time / int8_time - 1) * 100:.1f}%")
        
        # Create comparison image
        comparison = draw_comparison(image, fp16_detections, int8_detections)
        
        # Save comparison
        output_path = os.path.join(OUTPUT_DIR, f"comparison_{Path(img_path).stem}.jpg")
        cv2.imwrite(output_path, comparison)
        print(f"  Saved comparison to: {output_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    """Main function - chọn chế độ từ command line argument"""
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Get mode from command line
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
    else:
        mode = "run"  # Default mode
    
    onnx_file = "ONNX/yolov8n.onnx"
    fp16_engine_file = "ONNX/yolov8n.engine"
    int8_engine_file = "ONNX/yolov8n-int8.engine"
    input_shape = (1, 3, 640, 640)
    
    if mode == "build_fp16":
        # Build FP16 engine
        build_tensorrt_engine_fp16(onnx_file, fp16_engine_file)
    
    elif mode == "build_int8":
        # Build INT8 engine with calibration
        build_tensorrt_engine_int8(onnx_file, int8_engine_file, IMAGE_PATHS, input_shape)
    
    elif mode == "run_fp16":
        # Run inference with FP16 engine
        print("[INFO] Loading FP16 TensorRT engine...")
        detector = YOLOv8TensorRT(fp16_engine_file)
        run_inference(detector, IMAGE_PATHS, "FP16")
    
    elif mode == "run_int8":
        # Run inference with INT8 engine
        print("[INFO] Loading INT8 TensorRT engine...")
        detector = YOLOv8TensorRT(int8_engine_file)
        run_inference(detector, IMAGE_PATHS, "INT8")
    
    elif mode == "compare":
        # Compare FP16 vs INT8
        print("[INFO] Loading FP16 TensorRT engine...")
        fp16_detector = YOLOv8TensorRT(fp16_engine_file)
        print("[INFO] Loading INT8 TensorRT engine...")
        int8_detector = YOLOv8TensorRT(int8_engine_file)
        compare_engines(fp16_detector, int8_detector, IMAGE_PATHS)
    
    elif mode == "run":
        # Default: Run inference with FP16 engine
        print("[INFO] Loading FP16 TensorRT engine (default)...")
        detector = YOLOv8TensorRT(fp16_engine_file)
        run_inference(detector, IMAGE_PATHS, "FP16")
    
    else:
        print(f"[ERROR] Unknown mode: {mode}")
        print("\nUsage:")
        print("  python script.py build_fp16   - Build FP16 engine")
        print("  python script.py build_int8   - Build INT8 engine with calibration")
        print("  python script.py run_fp16     - Run inference with FP16 engine")
        print("  python script.py run_int8     - Run inference with INT8 engine")
        print("  python script.py compare      - Compare FP16 vs INT8 (side-by-side)")
        print("  python script.py              - Default (run_fp16)")
    
    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()