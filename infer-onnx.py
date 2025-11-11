import argparse
import cv2
import numpy as np
import onnxruntime as ort

# List of 80 class labels for COCO (can be trimmed/modified as needed)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left,
                             right, cv2.BORDER_CONSTANT, value=color)
    return img, (top, left)


def preprocess(img_path, input_width, input_height):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_height, img_width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lb, pad = letterbox(img_rgb, (input_width, input_height))
    img_norm = img_lb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_exp = np.expand_dims(img_chw, axis=0)
    return img_exp, pad, img, img_height, img_width


def draw_detections(img, box, score, class_id, color_palette, classes):
    x1, y1, w, h = box
    color = color_palette[class_id % len(color_palette)]
    color = tuple(int(c) for c in color)
    cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2)

    label = f"{classes[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x +
                  label_width, label_y + label_height), color, cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def postprocess(
    input_image,
    output,
    pad,
    input_height,
    input_width,
    orig_img_height,
    orig_img_width,
    conf_thres,
    iou_thres,
    classes=None,
    color_palette=None
):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    gain = min(input_height / orig_img_height, input_width / orig_img_width)
    outputs[:, 0] -= pad[1]
    outputs[:, 1] -= pad[0]

    for i in range(rows):
        cls_scores = outputs[i][4:]
        max_score = np.amax(cls_scores)
        if max_score >= conf_thres:
            class_id = np.argmax(cls_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) / gain)
            top = int((y - h / 2) / gain)
            width = int(w / gain)
            height = int(h / gain)
            class_ids.append(class_id)
            scores.append(float(max_score))
            boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    # cv2 4.8+ returns tuple if no detections
    if isinstance(indices, tuple) or len(indices) == 0:
        return input_image
    for idx in indices.flatten():
        draw_detections(
            input_image, boxes[idx], scores[idx], class_ids[idx], color_palette, classes)
    return input_image


if __name__ == "__main__":
    model = "ONNX/yolov8n.onnx"
    img = "person.jpg"
    input_width = 640
    input_height = 640
    conf_thres = 0.5
    iou_thres = 0.5

    color_palette = np.random.uniform(0, 255, size=(len(COCO_CLASSES), 3))

    # ONNX Inference session
    session = ort.InferenceSession(model, providers=[
        "CPUExecutionProvider"
    ])
    input_shape = session.get_inputs()[0].shape
    # input_shape may contain None or strings for dynamic dims; coerce to int with fallback
    def _safe_dim(x, fallback):
        try:
            return int(x)
        except Exception:
            return fallback

    input_width = _safe_dim(input_shape[2], input_width)
    input_height = _safe_dim(input_shape[3], input_height)

    # Preprocess
    img_data, pad, orig_img, orig_img_height, orig_img_width = preprocess(
        img, input_width, input_height)

    # Run inference (use the model's input name)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_data})

    # Postprocess
    out_img = postprocess(
        orig_img,
        outputs,
        pad,
        input_height,
        input_width,
        orig_img_height,
        orig_img_width,
        conf_thres,
        iou_thres,
        classes=COCO_CLASSES,
        color_palette=color_palette,
    )

    # Save result to file instead of showing GUI (headless-friendly)
    out_path = "out_infer.jpg"
    cv2.imwrite(out_path, out_img)
    print(f"Saved output to: {out_path}")
