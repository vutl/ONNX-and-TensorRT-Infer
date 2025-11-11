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
trt_path_onnx = '/home/atin/vutl/ONNX/yolov8n_fromonnx.engine' # TensorRT từ .onnx

image_paths = [
    '/home/atin/vutl/ONNX/person.jpg',
    '/home/atin/vutl/ONNX/car.jpg',
    '/home/atin/vutl/ONNX/bus.jpg'
]

os.makedirs('./output_imgs', exist_ok=True)

########### 1. Export models###########

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

# EXPORT TENSORRT từ .onnx BẰNG trtexec hoặc catch lỗi
if not os.path.exists(trt_path_onnx):
    print('[*] Export TensorRT .engine từ .onnx bằng trtexec...')
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path_onnx}",
        "--fp16",
        "--workspace=4096",
        "--shapes=images:1x3x640x640"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(result.stdout)
        if os.path.exists(trt_path_onnx):
            print('[*] File TensorRT (.onnx to engine) đã xuất:', trt_path_onnx)
        else:
            print("[!] LỖI EXPORT: Không tạo được file TensorRT (.onnx to engine).")
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"[!] LỖI khi chạy trtexec: {e}")
else:
    print('[*] File TensorRT (.onnx to engine) đã có:', trt_path_onnx)

########### 2. Tiện ích I/O ###########

def load_image_for_onnx(path, imgsz=640):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img

def plot_boxes(img_path, boxes, save_path):
    COLORS = [(0,255,0), (0,0,255), (255,0,0), (255,255,0), (255,128,0), (255,0,255)]
    img = cv2.imread(img_path)
    for box in boxes:
        x1, y1, x2, y2, score, clsid = box
        color = COLORS[int(clsid)%len(COLORS)]
        label = f"{int(clsid)}:{score:.2f}"
        cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        cv2.putText(img, label, (int(x1),int(y1)-5), 0, 1, color, 2)
    cv2.imwrite(save_path, img)

def analyze_boxes(boxes):
    if not hasattr(boxes, 'shape') or len(boxes)==0 or (hasattr(boxes, 'shape') and boxes.shape[0]==0):
        return 0, set(), (0,)
    classes = set(np.unique(boxes[:, 5].astype(int))) if boxes.shape[1] > 5 else set()
    return len(boxes), classes, boxes.shape

########### 3. CHẠY INFER ###########

results = []
pipelines = [
    ("Ultralytics YOLOv8n.pt", lambda: YOLO(pt_path) if os.path.exists(pt_path) else None),
    ("ONNX CPU", lambda: ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']) if os.path.exists(onnx_path) else None),
    ("ONNX GPU", lambda: ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider']) if os.path.exists(onnx_path) else None),
    ("TensorRT (.pt to engine)", lambda: YOLO(trt_path_u) if os.path.exists(trt_path_u) else None),
    ("TensorRT (onnx to engine)", lambda: YOLO(trt_path_onnx) if os.path.exists(trt_path_onnx) else None),
]

for name, loader in pipelines:
    print(f"\n[*] BENCHMARK: {name}")
    model = loader()
    if model is None:
        print(f"[!] Pipeline '{name}' - model not found. HÃY XEM LẠI FILE EXPORT!")
        continue

    if "ONNX" in name:
        # In provider info
        if hasattr(model, "get_providers"):
            print("  Providers:", model.get_providers())
            print("  Provider options:", model.get_provider_options())

    times, total_dets, total_classes = [], [], set()
    out_shapes = []
    for i, img_path in enumerate(image_paths):
        # Predict + lưu bbox
        if "Ultralytics" in name or "TensorRT" in name:
            t0 = time.time()
            res = model(img_path)
            t = time.time() - t0
            boxes = res[0].boxes.data.cpu().numpy()
        elif "ONNX" in name:
            img = load_image_for_onnx(img_path)
            t0 = time.time()
            outs = model.run(None, {"images": img})
            t = time.time() - t0
            # Nếu output shape chưa phải (N,6), skip vẽ bbox
            if hasattr(outs[0], 'shape') and (outs[0].ndim == 2 and outs[0].shape[1]==6):
                boxes = outs[0]
            else:
                boxes = np.zeros((0,6), dtype=np.float32)
        else:
            boxes = np.zeros((0,6), dtype=np.float32); t=0

        n_det, cls_set, shape = analyze_boxes(boxes)
        times.append(t)
        total_dets.append(n_det)
        total_classes.update(cls_set)
        out_shapes.append(shape)
        # Lưu bounding box (nếu có)
        img_savepath = f"./output_imgs/{os.path.basename(img_path).split('.')[0]}_{name.replace(' ','_').replace('(','').replace(')','')}.jpg"
        if n_det>0:
            plot_boxes(img_path, boxes, img_savepath)
            print(f"  -> Output img: {img_savepath}")

    results.append((name, np.mean(times), np.sum(total_dets), len(total_classes), out_shapes))

########### 4. In bảng tổng kết ###########
print(f"\n{'Backend':32s} {'AvgTime(s)':>12s} {'TotalDet':>10s} {'NumClass':>10s} {'Shapes'}")
for name, t, ndet, ncls, shapes in results:
    print(f"{name:32s} {t:12.4f} {ndet:10d} {ncls:10d} {shapes}")

print('\nẢnh kết quả bounding box đã lưu trong ./output_imgs/')

