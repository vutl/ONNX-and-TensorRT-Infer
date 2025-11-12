import os
import time
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort

pt_path = '/home/atin/vutl/ONNX/yolov8n.pt'
onnx_path = '/home/atin/vutl/ONNX/yolov8n.onnx'
trt_path_u = '/home/atin/vutl/ONNX/yolov8n.engine'             # TensorRT từ .pt
# trt_path_onnx removed: we no longer build or benchmark ONNX->TensorRT engine

image_paths = [
    '/home/atin/vutl/ONNX/person.jpg',
    '/home/atin/vutl/ONNX/car.jpg',
    '/home/atin/vutl/ONNX/bus.jpg'
]

os.makedirs('./output_imgs', exist_ok=True)

########### 1. Export models ###########

if not os.path.exists(onnx_path):
    print('[*] Export ONNX từ .pt...')
    YOLO(pt_path).export(format='onnx', imgsz=640, simplify=True)
    print('[*] File ONNX đã xuất:', onnx_path)
else:
    print('[*] File ONNX đã có:', onnx_path)

if not os.path.exists(trt_path_u):
    print('[*] Export TensorRT .engine trực tiếp từ .pt...')
    YOLO(pt_path).export(format='engine', imgsz=640)
    print('[*] File TensorRT (.pt to engine) đã xuất:', trt_path_u)
else:
    print('[*] File TensorRT (.pt to engine) đã có:', trt_path_u)

# Note: ONNX->TensorRT conversion removed to simplify benchmarking pipeline.

########### 2. Tiện ích I/O ###########

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # height, width
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (top, left)

def preprocess_for_onnx_inference(img_path, input_width=640, input_height=640):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    orig_h, orig_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lb, pad = letterbox(img_rgb, (input_width, input_height))
    img_norm = img_lb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_exp = np.expand_dims(img_chw, axis=0)
    return img_exp, pad, (orig_h, orig_w)

def plot_boxes(img_path, boxes, save_path):
    COLORS = [(0,255,0), (0,0,255), (255,0,0), (255,255,0), (255,128,0), (255,0,255)]
    img = cv2.imread(img_path)
    for box in boxes:
        x1, y1, x2, y2, score, clsid = box
        color = COLORS[int(clsid) % len(COLORS)]
        label = f"{int(clsid)}:{score:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), int(y1)-5), 0, 1, color, 2)
    cv2.imwrite(save_path, img)

def analyze_boxes(boxes):
    if not hasattr(boxes, 'shape') or len(boxes) == 0 or (hasattr(boxes, 'shape') and boxes.shape[0] == 0):
        return 0, set(), (0,)
    classes = set(np.unique(boxes[:,5].astype(int))) if boxes.shape[1] > 5 else set()
    return len(boxes), classes, boxes.shape

########### 3. Chạy infer ###########

results = []
pipelines = [
    ("Ultralytics YOLOv8n.pt", lambda: YOLO(pt_path) if os.path.exists(pt_path) else None),
    ("ONNX CPU", lambda: ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']) if os.path.exists(onnx_path) else None),
    ("ONNX GPU", lambda: ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider']) if os.path.exists(onnx_path) else None),
    ("TensorRT (.pt to engine)", lambda: YOLO(trt_path_u) if os.path.exists(trt_path_u) else None),
]

for name, loader in pipelines:
    print(f"\n[*] BENCHMARK: {name}")
    model = loader()
    if model is None:
        print(f"[!] Pipeline '{name}' - model not found. HÃY XEM LẠI FILE EXPORT!")
        continue

    if "ONNX" in name:
        if hasattr(model, "get_providers"):
            print("  Providers:", model.get_providers())
            print("  Provider options:", model.get_provider_options())

    times, total_dets, total_classes = [], [], set()
    out_shapes = []
    for i, img_path in enumerate(image_paths):
        if "Ultralytics" in name or "TensorRT" in name:
            t0 = time.time()
            res = model(img_path)
            t = time.time() - t0
            boxes = res[0].boxes.data.cpu().numpy()
        elif "ONNX" in name:
            # Preprocess and keep pad/original size for correct rescaling
            img, pad, (orig_h, orig_w) = preprocess_for_onnx_inference(img_path, 640, 640)
            t0 = time.time()
            # Use the model's input name to avoid mismatch
            try:
                input_name = model.get_inputs()[0].name
            except Exception:
                input_name = 'images'
            outs = model.run(None, {input_name: img})
            t = time.time() - t0

            # Normalize outputs to a (N, >=6) array if possible and convert to
            # [x1,y1,x2,y2,score,class] expected by plot_boxes/analyze_boxes
            arr = np.squeeze(outs[0])
            boxes_list = []
            conf_thres = 0.5
            if arr.ndim == 1:
                # no detections
                boxes = np.zeros((0, 6), dtype=np.float32)
            else:
                # arr expected to be (N, C) where C >= 6 (x,y,w,h, ...scores)
                # If arr has shape (C, N) due to unexpected transpose, try transpose
                if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                    # Heuristic: transpose when rows are smaller than cols
                    arr = arr.T

                # compute rescaling gain and remove padding (pad is (top,left))
                gain = min(640 / orig_h, 640 / orig_w)
                pad_top, pad_left = pad

                for row in arr:
                    if row.shape[0] < 5:
                        continue
                    x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    # class scores start at index 4 or 5 depending on export; handle both
                    if row.shape[0] > 5:
                        cls_scores = row[4:]
                        class_id = int(np.argmax(cls_scores))
                        score = float(np.max(cls_scores))
                    else:
                        # fallback: treat index 4 as score and class 0
                        score = float(row[4])
                        class_id = 0

                    if score < conf_thres:
                        continue

                    # remove pad offset then rescale to original image
                    x = x - pad_left
                    y = y - pad_top
                    left = int((x - w / 2) / gain)
                    top = int((y - h / 2) / gain)
                    right = int((x + w / 2) / gain)
                    bottom = int((y + h / 2) / gain)
                    boxes_list.append([left, top, right, bottom, score, class_id])

                # Apply class-wise NMS to remove duplicate detections
                final_boxes = []
                nms_iou_thres = 0.45
                conf_thres = 0.5
                if boxes_list:
                    boxes_arr = np.array(boxes_list, dtype=np.float32)
                    # boxes_arr columns: x1, y1, x2, y2, score, class_id
                    for cls in np.unique(boxes_arr[:, 5].astype(int)):
                        cls_mask = boxes_arr[:, 5].astype(int) == int(cls)
                        cls_boxes = boxes_arr[cls_mask]
                        # convert to x,y,w,h for NMS
                        xywh = []
                        scores = []
                        for b in cls_boxes:
                            x1, y1, x2, y2, score, _ = b
                            w = max(0, x2 - x1)
                            h = max(0, y2 - y1)
                            xywh.append([int(x1), int(y1), int(w), int(h)])
                            scores.append(float(score))

                        # run NMS
                        try:
                            idxs = cv2.dnn.NMSBoxes(xywh, scores, conf_thres, nms_iou_thres)
                        except Exception:
                            # fallback: keep all
                            idxs = tuple(range(len(xywh)))

                        if isinstance(idxs, tuple):
                            # OpenCV may return tuple when no detections
                            selected = list(idxs)
                        else:
                            selected = idxs.flatten().tolist() if hasattr(idxs, 'flatten') else [int(idxs)]

                        for si in selected:
                            final_boxes.append(cls_boxes[si].tolist())

                    boxes = np.array(final_boxes, dtype=np.float32) if final_boxes else np.zeros((0, 6), dtype=np.float32)
                else:
                    boxes = np.zeros((0, 6), dtype=np.float32)
        else:
            boxes = np.zeros((0,6), dtype=np.float32)
            t = 0

        n_det, cls_set, shape = analyze_boxes(boxes)
        times.append(t)
        total_dets.append(n_det)
        total_classes.update(cls_set)
        out_shapes.append(shape)

        img_savepath = f"./output_imgs/{os.path.basename(img_path).split('.')[0]}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
        if n_det > 0:
            plot_boxes(img_path, boxes, img_savepath)
            print(f"  -> Output img: {img_savepath}")

    results.append((name, np.mean(times), np.sum(total_dets), len(total_classes), out_shapes))

########### 4. In bảng tổng kết ###########
print(f"\n{'Backend':32s} {'AvgTime(s)':>12s} {'TotalDet':>10s} {'NumClass':>10s} {'Shapes'}")
for name, t, ndet, ncls, shapes in results:
    print(f"{name:32s} {t:12.4f} {ndet:10d} {ncls:10d} {shapes}")

print('\nẢnh kết quả bounding box đã lưu trong ./output_imgs/')
