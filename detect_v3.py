import os, time, json
from datetime import datetime
import cv2
import numpy as np
import requests
from ultralytics import YOLO

# ---------------- Config ----------------
MODEL_NAME = "yolov3u.pt"  # small upgrade over yolov3.pt; works in Ultralytics
CONF_THRES = 0.5
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/vehicles/vehicles")  # FastAPI POST
POST_EVERY_N_FRAMES = 10
SAVE_ANNOTATED = True
OUT_VIDEO = "output/annotated.mp4"  # set None to disable
SHOW_PREPROC = False  # set True to view preprocessed frame

vehicle_classes = {"car", "truck", "bus", "motorbike"}
emergency_keywords = {"ambulance", "fire truck", "police car"}  # model may not have these labels

# Define lanes (x1,y1,x2,y2) - adjust to your intersection image
lanes = {
    "North": (0, 0, 480, 720),
    "East": (480, 0, 960, 720),
    "South": (960, 0, 1440, 720),
    "West": (1440, 0, 1920, 720)
}

# -------------- Utilities ---------------
def auto_gamma(img, target=140.0):
    # estimate brightness from V channel and compute gamma
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_mean = float(np.mean(hsv[..., 2]))
    if v_mean < 1e-6:
        return img
    gamma = max(0.4, min(3.0, np.log(target/255.0) / np.log(v_mean/255.0) * -1.0))
    inv = 1.0 / gamma
    table = np.array([(i/255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def clahe_on_v(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    hsv[...,2] = clahe.apply(hsv[...,2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def grayworld_white_balance(img):
    # simple white balance that helps under warm dawn lighting
    result = img.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(result[:,:,0]), np.mean(result[:,:,1]), np.mean(result[:,:,2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    scale_b, scale_g, scale_r = avg_gray/(avg_b+1e-6), avg_gray/(avg_g+1e-6), avg_gray/(avg_r+1e-6)
    result[:,:,0] *= scale_b
    result[:,:,1] *= scale_g
    result[:,:,2] *= scale_r
    return np.clip(result, 0, 255).astype(np.uint8)

def light_denoise(img):
    # very mild to avoid blurring edges
    return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)

def preprocess(frame):
    # Pipeline tuned for dawn/low-light
    f = frame
    f = auto_gamma(f, target=150)
    f = clahe_on_v(f)
    f = grayworld_white_balance(f)
    f = light_denoise(f)
    return f

def point_in_lane(cx, cy, rect):
    x1,y1,x2,y2 = rect
    return x1 <= cx <= x2 and y1 <= cy <= y2

# -------------- Model -------------------
model = YOLO(MODEL_NAME)  # downloads if not present

# -------------- Detection ---------------
def detect_vehicles(image_bgr):
    # Preprocess
    pre = preprocess(image_bgr)

    # Inference
    r = model(pre, verbose=False)[0]

    lane_counts = {k: 0 for k in lanes}
    total = 0
    emergency = False

    # Draw lane rectangles (optional guide)
    for nm, (x1,y1,x2,y2) in lanes.items():
        cv2.rectangle(image_bgr, (x1,y1), (x2,y2), (60,60,60), 1)
        cv2.putText(image_bgr, nm, (x1+6, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    # Iterate detections
    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        c = int(cls.item())
        label = model.names[c]
        score = float(conf.item())
        if label not in vehicle_classes or score < CONF_THRES:
            continue

        total += 1
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1+x2)//2, (y1+y2)//2

        for nm, rect in lanes.items():
            if point_in_lane(cx, cy, rect):
                lane_counts[nm] += 1
                break

        if any(k in label.lower() for k in emergency_keywords):
            emergency = True

        cv2.rectangle(image_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.circle(image_bgr, (cx,cy), 3, (0,255,0), -1)
        cv2.putText(image_bgr, f"{label} {score:.2f}", (x1, max(20,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Extra emergency cue from flashing reds (very heuristic)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, (0,120,70), (10,255,255))
    if cv2.countNonZero(mask_red) > 1500:
        emergency = True

    # HUD
    cv2.putText(image_bgr, f"Total: {total}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    y = 60
    for nm, cnt in lane_counts.items():
        cv2.putText(image_bgr, f"{nm}: {cnt}", (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        y += 26
    if emergency:
        cv2.putText(image_bgr, "EMERGENCY DETECTED", (10,y+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)

    return image_bgr, pre if SHOW_PREPROC else None, total, lane_counts, emergency

# -------------- Runner ------------------
def main():
    os.makedirs("output", exist_ok=True)
    cap = cv2.VideoCapture("samples/video6.mp4")  # or 0 for webcam
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer = None
    if SAVE_ANNOTATED and OUT_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        annotated, pre, total, lane_counts, emergency = detect_vehicles(frame)

        # Show
        if SHOW_PREPROC and pre is not None:
            combo = np.hstack([cv2.resize(pre, (width//2, height)), cv2.resize(annotated, (width//2, height))])
            cv2.imshow("Preproc | Annotated", combo)
        else:
            cv2.imshow("Vehicle & Lane Detection", annotated)

        if writer: writer.write(annotated)

        # Log
        print(f"{datetime.now().isoformat(timespec='seconds')}  total={total}  lanes={lane_counts}  emergency={emergency}")

        # Send to backend every N frames
        if (frame_idx % POST_EVERY_N_FRAMES) == 0:
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "lane_counts": lane_counts,
                "total": total,
                "emergency": emergency
            }
            try:
                requests.post(API_URL, json=payload, timeout=1.5)
            except Exception:
                pass  # don't crash if backend is down

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
