import cv2
import numpy as np
import onnxruntime as ort

# ---------------- Letterbox ----------------
def letterbox(img, new_shape=(640, 640), color=(114,114,114)):
    h, w = img.shape[:2]
    new_w, new_h = new_shape
    scale = min(new_w/w, new_h/h)
    scaled_w, scaled_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (scaled_w, scaled_h))
    top = (new_h - scaled_h) // 2
    bottom = new_h - scaled_h - top
    left = (new_w - scaled_w) // 2
    right = new_w - scaled_w - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, scale, (left, top)

# ---------------- Decode Output ----------------

def decode_output(output, anchors, stride, scale, pad, nc=1):

    B, C, H, W = output.shape
    na = len(anchors)  # 3
    no = 5 + nc + 10   # xywh + obj + cls + 5kp*2

    # reshape → (B, na, no, H, W)
    out = output.reshape(B, na, no, H, W)

    # (B,na,H,W,no)
    out = out.transpose(0, 1, 3, 4, 2)

    boxes_all = []

    grid_x = np.arange(W)[None, :].repeat(H, axis=0)  # (H,W)
    grid_y = np.arange(H)[:, None].repeat(W, axis=1)  # (H,W)

    for b in range(B):
        for i, (aw, ah) in enumerate(anchors):
            pred = out[b, i]         # (H, W, no)
            cx = pred[..., 0]        # (H,W)
            cy = pred[..., 1]       
            bw = pred[..., 2]
            bh = pred[..., 3]
            obj = pred[..., 4]      
            cls_p = pred[..., 15:15+nc]   # (H,W,nc)
          
            x = (cx * 2 - 0.5 + grid_x) * stride
            y = (cy * 2 - 0.5 + grid_y) * stride
            w_box = (bw * 2) ** 2 * aw
            h_box = (bh * 2) ** 2 * ah

            conf = obj[..., None] * cls_p    
            # ----- 关键点 -----
            kps_list = []
            kp_start = 5

            for j in range(5):
                kx = pred[..., kp_start + j * 2] * aw + (grid_x  * stride)
                ky = pred[..., kp_start + j * 2 + 1] * ah + (grid_y *stride)
                kxy = np.stack([kx, ky], axis=-1)  # (H,W,2)
                kps_list.append(kxy)
      
            kps = np.stack(kps_list, axis=-2)
            x = (x - pad[0]) / scale
            y = (y - pad[1]) / scale
            w_box /= scale
            h_box /= scale
            kps = (kps - np.array(pad)) / scale

            for cls_i in range(nc):
                conf_map = conf[..., cls_i]   # (H,W)
                mask = conf_map > 0

                if np.any(mask):
                    box = np.concatenate([
                        x[..., None],
                        y[..., None],
                        w_box[..., None],
                        h_box[..., None],
                        conf_map[..., None],           
                        np.full_like(conf_map[..., None], cls_i),  
                        kps.reshape(H, W, -1)         
                    ], axis=-1)

                    boxes_all.append(box[mask])

    if len(boxes_all) == 0:
        return np.zeros((0, 16+1))

    return np.concatenate(boxes_all, axis=0)


def nms(boxes, conf_thresh=0.3, iou_thresh=0.5):
    """
    boxes: [N, 16] -> xywh, conf, 5 keypoints..., cls
    返回保留的 boxes
    """
    if boxes.shape[0] == 0:
        return np.array([])

    keep_boxes = []

    for cls in np.unique(boxes[:, 5]):  # 按类别做 NMS
        cls_mask = boxes[:, 5] == cls
        cls_boxes = boxes[cls_mask]
        cls_boxes = cls_boxes[cls_boxes[:, 4] > conf_thresh]

        if len(cls_boxes) == 0:
            continue

        x = cls_boxes[:, 0]
        y = cls_boxes[:, 1]
        w = cls_boxes[:, 2]
        h = cls_boxes[:, 3]
        scores = cls_boxes[:, 4]

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        idxs = scores.argsort()[::-1]

        while len(idxs) > 0:
            i = idxs[0]
            keep_boxes.append(cls_boxes[i])

            if len(idxs) == 1:
                break

            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter = inter_w * inter_h

            union = (w[i]*h[i] + w[idxs[1:]]*h[idxs[1:]] - inter)
            iou = inter / union

            idxs = idxs[1:][iou < iou_thresh]

    if len(keep_boxes) == 0:
        return np.array([])
    return np.stack(keep_boxes)

# ---------------- 绘制 ----------------
def draw_boxes(img, boxes):
    for b in boxes:
        print("所有:",b)
        x, y, w, h, conf = b[:5]
        cls = b[5]
        kps = b[6:16].reshape(5,2)
        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
        cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
        for kp in kps:
            cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (0,0,255), -1)

        cv2.putText(img, f"{conf:.3f}:{    int(cls)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
       
    return img


anchors_all = [
    [(4,5),(8,10),(13,16)],   # P3
    [(23,29),(43,55),(73,105)],   # P4
    [(146,217),(231,300),(335,433)]  # P5
]
stride_list = [8, 16, 32]  # 每层特征对应 stride

nc = 1  # 类别数，单类别=1，多类别>1

session = ort.InferenceSession("yolov5n_face.onnx")
img = cv2.imread("images/0_Parade_marchingband_1_593.jpg")
img_input, scale, pad = letterbox(img, (640,640))
img_input = img_input.transpose(2,0,1)[np.newaxis,...]/255.0
img_input = img_input.astype(np.float32)

outputs = session.run(None, {session.get_inputs()[0].name: img_input})

all_boxes = []
for out, anchors, s in zip(outputs, anchors_all, stride_list):
    boxes = decode_output(np.array(out), anchors, s, scale, pad, nc=nc)
    all_boxes.append(boxes)


if len(all_boxes) > 0:
    boxes = np.concatenate(all_boxes, axis=0)
    print("boxes.shape", boxes.shape)
    boxes = nms(boxes, conf_thresh=0.3, iou_thresh=0.5)
    print("boxes.shape", boxes.shape)
else:
    boxes = np.array([])

img_draw = draw_boxes(img.copy(), boxes)
cv2.imwrite("result.jpg", img_draw)

