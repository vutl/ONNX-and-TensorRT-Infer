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
FP16_ENGINE_PATH = "weights/yolov8n_fp16.engine"
INT8_ENGINE_PATH = "weights/yolov8n_int8.engine"

IMAGE_PATHS = [
    "/home/atin-tts-1/vutl/COCO-SMALL-3/train/images/img_000029.jpg",
    "/home/atin-tts-1/vutl/COCO-SMALL-3/train/images/img_000038.jpg",
    "/home/atin-tts-1/vutl/COCO-SMALL-3/train/images/img_000035.jpg",
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
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time

CONF_THRESH = 0.25
IOU_THRESH = 0.5
INPUT_SIZE = 640


class YOLOv8TensorRT:
    def __init__(self, engine_path):
        # --- Load engine bytes ---
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        # --- Create runtime + engine ---
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        # --- Tensor names ---
        self.input_name = "images"
        self.output_name = "output0"

        # --- Print raw shapes (-1,3,-1,-1) ---
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape_raw = self.engine.get_tensor_shape(self.output_name)

        print(f"[INFO] Input shape (engine):  {self.input_shape}")
        print(f"[INFO] Output shape (engine): {self.output_shape_raw}")

        self.stream = cuda.Stream()

    # -----------------------------------------------------------
    def infer(self, image, batch_size=1):
        # Preprocess cho batch 1 → (1,3,640,640)
        input_data, scale_h, scale_w, pad_h, pad_w = preprocess_static(
            image, (1, 3, INPUT_SIZE, INPUT_SIZE)
        )

        # Nếu batch > 1, lặp lại ảnh
        if batch_size > 1:
            input_data = np.repeat(input_data, batch_size, axis=0)

        # --- Set dynamic shape ---
        self.context.set_input_shape(self.input_name, input_data.shape)

        # --- Lấy output shape sau khi set ---
        output_shape = self.context.get_tensor_shape(self.output_name)
        print(f"[INFO] Runtime output shape resolved: {output_shape}")

        # --- Allocate device memory ---
        d_input = cuda.mem_alloc(input_data.nbytes)
        output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        # --- Copy input ---
        cuda.memcpy_htod_async(d_input, input_data, self.stream)

        # --- Bindings ---
        self.context.set_tensor_address(self.input_name, int(d_input))
        self.context.set_tensor_address(self.output_name, int(d_output))

        # --- Execute ---
        start = time.time()
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.stream.synchronize()
        infer_time = (time.time() - start) * 1000

        # Free GPU memory
        d_input.free()
        d_output.free()

        # Postprocess cho batch=1 (demo)
        detections = self.postprocess(
            output[0], image.shape[0], image.shape[1], scale_h, scale_w, pad_h, pad_w
        )
        return detections, infer_time

    # -----------------------------------------------------------
    def postprocess(self, output, orig_h, orig_w, scale_h, scale_w, pad_h, pad_w):
        # YOLOv8 format: (84,8400) → transpose → (8400,84)
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

        # Convert xywh → xyxy
        xc, yc, w, h = boxes.T
        xc = (xc - pad_w) / scale_w
        yc = (yc - pad_h) / scale_h
        w /= scale_w
        h /= scale_h

        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2

        x1 = np.clip(x1, 0, orig_w - 1)
        y1 = np.clip(y1, 0, orig_h - 1)
        x2 = np.clip(x2, 0, orig_w - 1)
        y2 = np.clip(y2, 0, orig_h - 1)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = self.nms(boxes_xyxy, scores, IOU_THRESH)

        return [
            (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
             float(scores[i]), int(classes[i]))
            for i in keep
        ]

    # NMS -------------------------------------------------------
    @staticmethod
    def nms(boxes, scores, iou_thresh):
        if len(boxes) == 0:
            return []

        idxs = np.argsort(scores)[::-1]
        keep = []

        while len(idxs) > 0:
            cur = idxs[0]
            keep.append(cur)
            if len(idxs) == 1:
                break

            ious = YOLOv8TensorRT.calc_iou(boxes[cur], boxes[idxs[1:]])
            idxs = idxs[1:][ious < iou_thresh]

        return keep

    @staticmethod
    def calc_iou(box, boxes):
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
