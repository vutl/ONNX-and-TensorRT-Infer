import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path

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

# COCO class
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
# TensorRT Engine Loader
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
        print(f"[INFO] Input shape: {self.input_shape}")
        print(f"[INFO] Output shape: {self.output_shape}")
        # Tạo 1 stream dùng chung
        self.stream = cuda.Stream()

    def infer(self, image):
        # Preprocessing
        input_data, scale_h, scale_w, pad_h, pad_w = self.preprocess(image)
        # Input buffer
        d_input = cuda.mem_alloc(input_data.nbytes)
        # Output buffer
        output_size = np.prod(self.output_shape)
        output = np.empty(self.output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)  # Đúng số bytes output
        # Copy input to GPU
        cuda.memcpy_htod_async(d_input, input_data, self.stream)
        # Set tensor address
        self.context.set_tensor_address("images", int(d_input))
        self.context.set_tensor_address("output0", int(d_output))
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        # Copy output from GPU về host
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.stream.synchronize()  # Đảm bảo xong rồi mới dùng output
        d_input.free()
        d_output.free()
        # Postprocess
        detections = self.postprocess(
            output, image.shape[0], image.shape[1], scale_h, scale_w, pad_h, pad_w
        )
        return detections
 
    def preprocess(self, image):
        """
        Preprocess image for YOLOv8
        - Letterbox resize (maintain aspect ratio)
        - Normalize to [0, 1]
        - Convert BGR -> RGB, HWC -> CHW
        
        Returns:
            input_data: (1, 3, 640, 640) in float32
            scale_h, scale_w, pad_h, pad_w: for postprocessing
        """
        h, w = image.shape[:2]
        target_size = self.input_shape[2]  # 640
        
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
        
        # Extract predictions
        predictions = output  # (8400, 84)
        
        # Get max class confidence for each prediction
        class_scores = predictions[:, 4:]  # (8400, 80)
        max_scores = np.max(class_scores, axis=1)  # (8400,)
        class_ids = np.argmax(class_scores, axis=1)  # (8400,)
        
        # Filter by confidence threshold
        mask = max_scores >= CONF_THRESH
        
        # Extract valid predictions
        valid_boxes = predictions[mask, :4]  # (N, 4) - xywh format
        valid_scores = max_scores[mask]  # (N,)
        valid_classes = class_ids[mask]  # (N,)
        
        # Convert from model coordinates to original image coordinates
        # xywh format
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
        if len(valid_scores) > 0:
            # Convert to xyxy format for NMS
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
        
        return []
    
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
        
        # Sort by score
        sorted_idx = np.argsort(scores)[::-1]
        
        keep_idx = []
        while len(sorted_idx) > 0:
            # Keep the box with highest score
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
# Visualization
# ============================================================================
def draw_boxes(image, detections):
    """Draw bounding boxes on image"""
    result = image.copy()
    
    for x1, y1, x2, y2, conf, class_id in detections:
        # Convert to int
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
    
    return result


# ============================================================================
# Main
# ============================================================================
def main():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print("[INFO] Loading TensorRT engine...")
    detector = YOLOv8TensorRT(ENGINE_PATH)
    
    for img_path in IMAGE_PATHS:
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            continue
        
        print(f"\n[INFO] Processing: {img_path}")
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Failed to read image: {img_path}")
            continue
        
        h, w = image.shape[:2]
        print(f"  Image size: {w}x{h}")
        
        # Run inference
        detections = detector.infer(image)
        print(f"  Detections: {len(detections)}")
        
        for i, (x1, y1, x2, y2, conf, class_id) in enumerate(detections):
            print(f"    [{i}] {COCO_CLASSES[class_id]}: conf={conf:.3f}, box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        # Draw boxes
        result = draw_boxes(image, detections)
        
        # Save output
        output_path = os.path.join(OUTPUT_DIR, f"result_{Path(img_path).stem}.jpg")
        cv2.imwrite(output_path, result)
        print(f"  Saved to: {output_path}")
    
    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()