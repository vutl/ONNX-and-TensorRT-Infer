import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import sys
import time
from collections import Counter


# ============================================================================
# Configuration
# ============================================================================
FP16_ENGINE_PATH = "ONNX/yolov8n.engine"
INT8_ENGINE_PATH = "ONNX/yolov8n-int8.engine"

IMAGE_PATHS = [
    "ONNX/inference/bus.jpg",
    "ONNX/inference/car.jpg",
    "ONNX/inference/person.jpg"
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

COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


# ============================================================================
# Preprocess (dùng cho inference)
# ============================================================================
def preprocess_static(image, input_shape):
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
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    input_data = np.expand_dims(chw, axis=0)
    input_data = np.ascontiguousarray(input_data, dtype=np.float32)
    
    return input_data, scale, scale, pad_top, pad_left


# ============================================================================
# TensorRT Engine Loader + Inference
# ============================================================================
class YOLOv8TensorRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_tensor_shape("images")
        self.input_dtype = self.engine.get_tensor_dtype("images")
        self.output_shape = self.engine.get_tensor_shape("output0")
        
        print(f"[INFO] Loaded engine: {engine_path}")
        print(f"[INFO] Input shape: {self.input_shape}")
        print(f"[INFO] Output shape: {self.output_shape}")
        
        self.stream = cuda.Stream()

    def infer(self, image):
        input_data, scale_h, scale_w, pad_h, pad_w = preprocess_static(image, self.input_shape)
        
        d_input = cuda.mem_alloc(input_data.nbytes)
        output = np.empty(self.output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)
        
        cuda.memcpy_htod_async(d_input, input_data, self.stream)
        self.context.set_tensor_address("images", int(d_input))
        self.context.set_tensor_address("output0", int(d_output))
        
        start_time = time.time()
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.stream.synchronize()
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000.0
        
        d_input.free()
        d_output.free()
        
        detections = self.postprocess(
            output, image.shape[0], image.shape[1], scale_h, scale_w, pad_h, pad_w
        )
        
        return detections, inference_time

    def postprocess(self, output, orig_h, orig_w, scale_h, scale_w, pad_h, pad_w):
        output = output[0]        # (84, 8400)
        output = output.T         # (8400, 84)
        
        class_scores = output[:, 4:]
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        mask = max_scores >= CONF_THRESH
        valid_boxes = output[mask, :4]
        valid_scores = max_scores[mask]
        valid_classes = class_ids[mask]
        
        if len(valid_scores) == 0:
            return []
        
        x_center = valid_boxes[:, 0]
        y_center = valid_boxes[:, 1]
        width = valid_boxes[:, 2]
        height = valid_boxes[:, 3]
        
        x_center = (x_center - pad_w) / scale_h
        y_center = (y_center - pad_h) / scale_w
        width = width / scale_h
        height = height / scale_w
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)
        
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
        if len(boxes) == 0:
            return []
        
        sorted_idx = np.argsort(scores)[::-1]
        keep_idx = []
        while len(sorted_idx) > 0:
            current_idx = sorted_idx[0]
            keep_idx.append(current_idx)
            if len(sorted_idx) == 1:
                break
            current_box = boxes[current_idx]
            remaining_boxes = boxes[sorted_idx[1:]]
            ious = YOLOv8TensorRT.calc_iou(current_box, remaining_boxes)
            keep_mask = ious < iou_threshold
            sorted_idx = sorted_idx[1:][keep_mask]
        return keep_idx

    @staticmethod
    def calc_iou(box, boxes):
        x1_min, y1_min, x1_max, y1_max = box
        x2_min = boxes[:, 0]
        y2_min = boxes[:, 1]
        x2_max = boxes[:, 2]
        y2_max = boxes[:, 3]
        
        inter_x_min = np.maximum(x1_min, x2_min)
        inter_y_min = np.maximum(y1_min, y2_min)
        inter_x_max = np.minimum(x1_max, x2_max)
        inter_y_max = np.minimum(y1_max, y2_max)
        
        inter_w = np.clip(inter_x_max - inter_x_min, 0, None)
        inter_h = np.clip(inter_y_max - inter_y_min, 0, None)
        inter_area = inter_w * inter_h
        
        box_area = (x1_max - x1_min) * (y1_max - y1_min)
        boxes_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box_area + boxes_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou


# ============================================================================
# Visualization
# ============================================================================
def draw_boxes(image, detections, title="Detection"):
    result = image.copy()
    for x1, y1, x2, y2, conf, class_id in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = tuple(map(int, COLORS[class_id]))
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
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
    
    cv2.putText(result, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return result


def draw_comparison(image, fp16_detections, int8_detections):
    fp16_result = draw_boxes(image.copy(), fp16_detections, f"FP16 ({len(fp16_detections)} detections)")
    int8_result = draw_boxes(image.copy(), int8_detections, f"INT8 ({len(int8_detections)} detections)")
    comparison = np.hstack([fp16_result, int8_result])
    return comparison


# ============================================================================
# Analysis helpers
# ============================================================================
def analyze_differences(fp16_dets, int8_dets):
    print("  [ANALYSIS] Detailed comparison FP16 vs INT8")
    if len(fp16_dets) == 0 or len(int8_dets) == 0:
        print("    One of the engines produced 0 detections, skip IoU/conf analysis.")
    else:
        ious = []
        conf_diffs = []
        same_cls_flags = []

        for (x1_t, y1_t, x2_t, y2_t, conf_t, cls_t) in int8_dets:
            best_iou = 0.0
            best_conf = None
            best_cls = None

            for (x1_r, y1_r, x2_r, y2_r, conf_r, cls_r) in fp16_dets:
                inter_x1 = max(x1_t, x1_r)
                inter_y1 = max(y1_t, y1_r)
                inter_x2 = min(x2_t, x2_r)
                inter_y2 = min(y2_t, y2_r)

                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                area_t = max(0.0, (x2_t - x1_t)) * max(0.0, (y2_t - y1_t))
                area_r = max(0.0, (x2_r - x1_r)) * max(0.0, (y2_r - y1_r))
                union = area_t + area_r - inter_area + 1e-6

                iou = inter_area / union

                if iou > best_iou:
                    best_iou = iou
                    best_conf = conf_r
                    best_cls = cls_r

            if best_conf is not None:
                ious.append(best_iou)
                conf_diffs.append(conf_t - best_conf)
                same_cls_flags.append(int(cls_t == best_cls))

        if ious:
            mean_iou = float(np.mean(ious))
            mean_abs_iou = float(np.mean(np.abs(ious)))
            mean_conf_diff = float(np.mean(conf_diffs))
            mean_abs_conf_diff = float(np.mean(np.abs(conf_diffs)))
            same_cls_ratio = sum(same_cls_flags) / len(same_cls_flags)

            print(f"    Mean IoU (INT8 vs FP16 best match): {mean_iou:.3f}")
            print(f"    Mean |IoU|: {mean_abs_iou:.3f}")
            print(f"    Mean Δconf (INT8 - FP16): {mean_conf_diff:.3f}")
            print(f"    Mean |Δconf|: {mean_abs_conf_diff:.3f}")
            print(f"    % same class (matched pairs): {same_cls_ratio*100:.1f}%")
        else:
            print("    No valid matches to compute IoU/conf differences.")

    fp16_cls = Counter([d[5] for d in fp16_dets])
    int8_cls = Counter([d[5] for d in int8_dets])

    print("    Class counts (FP16 vs INT8):")
    all_classes = sorted(set(fp16_cls.keys()) | set(int8_cls.keys()))
    for cls_id in all_classes:
        name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else str(cls_id)
        print(f"      {name:15s}: {fp16_cls[cls_id]:2d} vs {int8_cls[cls_id]:2d}")


# ============================================================================
# Inference helpers
# ============================================================================
def run_inference(detector, images, engine_name="Engine"):
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
        
        detections, infer_time = detector.infer(image)
        print(f"  Detections: {len(detections)}")
        print(f"  Inference time: {infer_time:.2f} ms")
        
        for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
            print(f"    [{i}] {COCO_CLASSES[class_id]}: conf={conf:.3f}, box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        result = draw_boxes(image, detections, f"{engine_name} ({len(detections)} detections)")
        output_path = os.path.join(OUTPUT_DIR, f"result_{engine_name}_{Path(img_path).stem}.jpg")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"  Saved to: {output_path}")


def compare_engines(fp16_detector, int8_detector, images):
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
        
        print("  Running FP16 inference...")
        fp16_dets, fp16_time = fp16_detector.infer(image)
        print(f"    FP16 detections: {len(fp16_dets)}, Time: {fp16_time:.2f} ms")
        for i, (_, _, _, _, conf, cls_id) in enumerate(fp16_dets):
            print(f"      FP16 [{i}] {COCO_CLASSES[cls_id]}: conf={conf:.3f}")
        
        print("  Running INT8 inference...")
        int8_dets, int8_time = int8_detector.infer(image)
        print(f"    INT8 detections: {len(int8_dets)}, Time: {int8_time:.2f} ms")
        for i, (_, _, _, _, conf, cls_id) in enumerate(int8_dets):
            print(f"      INT8 [{i}] {COCO_CLASSES[cls_id]}: conf={conf:.3f}")
        
        print(f"\n  [COMPARISON]")
        print(f"    Detection count difference: {abs(len(fp16_dets) - len(int8_dets))} boxes")
        if int8_time > 0:
            speedup = fp16_time / int8_time
            print(f"    Speed (FP16): {fp16_time:.2f} ms")
            print(f"    Speed (INT8): {int8_time:.2f} ms")
            print(f"    Speedup (FP16 / INT8): {speedup:.2f}x ({(speedup - 1) * 100:.1f}% faster)")
        else:
            print("    INT8 time is 0 ms? (check timing)")

        analyze_differences(fp16_dets, int8_dets)
        
        comparison = draw_comparison(image, fp16_dets, int8_dets)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"comparison_{Path(img_path).stem}.jpg")
        cv2.imwrite(output_path, comparison)
        print(f"  Saved comparison to: {output_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    """
    Usage:
      python infer_tensorrt.py run_fp16   -> chỉ chạy FP16
      python infer_tensorrt.py run_int8   -> chỉ chạy INT8
      python infer_tensorrt.py compare    -> so sánh FP16 vs INT8
      python infer_tensorrt.py            -> mặc định: run_fp16
    """
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
    else:
        mode = "run_fp16"
    
    if mode == "run_fp16":
        print("[INFO] Loading FP16 engine...")
        detector = YOLOv8TensorRT(FP16_ENGINE_PATH)
        run_inference(detector, IMAGE_PATHS, "FP16")
    
    elif mode == "run_int8":
        print("[INFO] Loading INT8 engine...")
        detector = YOLOv8TensorRT(INT8_ENGINE_PATH)
        run_inference(detector, IMAGE_PATHS, "INT8")
    
    elif mode == "compare":
        print("[INFO] Loading FP16 engine...")
        fp16_detector = YOLOv8TensorRT(FP16_ENGINE_PATH)
        print("[INFO] Loading INT8 engine...")
        int8_detector = YOLOv8TensorRT(INT8_ENGINE_PATH)
        compare_engines(fp16_detector, int8_detector, IMAGE_PATHS)
    
    else:
        print(f"[ERROR] Unknown mode: {mode}")
        print("  Modes: run_fp16 | run_int8 | compare")
    
    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
