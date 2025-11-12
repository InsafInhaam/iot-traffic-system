import threading
import time
import json
import os
from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, jsonify

# ---------- Config ----------
LANES_FILE = "lanes_config.json"
WEIGHTS = "yolov8n.pt"     # YOLO model weights
WEBCAM_INDEX = 0           # USB webcam index (0,1,...)
FPS = 5                    # Detection FPS (approx)
ROLLING_WINDOW = 6         # number of recent counts to average
MIN_GREEN = 8              # minimum green duration (seconds)
MAX_GREEN = 30             # maximum green duration (seconds)
API_PORT = 6000

# ---------- Globals shared between threads ----------
_current_counts = {}       # lane -> deque of recent counts
_current_signal = {"lane": None, "duration": MIN_GREEN, "counts": {}}
_lock = threading.Lock()

# ---------- Utilities ----------


def load_lanes(config_path=LANES_FILE):
    if not os.path.exists(config_path):
        print("No lanes_config.json found. Run calibration first.")
        return {}
    with open(config_path, "r") as f:
        data = json.load(f)
    # Expect top-level keys are filenames (we load the first/only one for live)
    # We'll just pick lanes for the key that matches the webcam demo image filename if present.
    # For live demo you should have created lanes for your demo image filename.
    # We'll just load the first key available if none match.
    if not data:
        return {}
    # pick first entry
    key = list(data.keys())[0]
    lanes = {}
    for name, pts in data[key].items():
        lanes[name] = np.array(pts, dtype=np.int32)
    print(
        f"[INFO] Loaded {len(lanes)} lanes from {config_path} (key='{key}').")
    return lanes


def point_in_polygon(pt, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.float32), tuple(pt), False) >= 0


def choose_signal_from_counts(avg_counts):
    # avg_counts: dict lane->float
    if not avg_counts:
        return None, MIN_GREEN
    # choose lane with max vehicles; if all zero, pick first
    lane = max(avg_counts.items(), key=lambda kv: kv[1])[0]
    max_val = avg_counts[lane]
    # map count to duration linearly (clamp)
    # assume reasonable max vehicles for scaling; scale_factor could be tuned
    SCALE_MAX = 8.0
    frac = min(max_val / SCALE_MAX, 1.0)
    duration = int(MIN_GREEN + frac * (MAX_GREEN - MIN_GREEN))
    return lane, duration

# ---------- Detection thread ----------


def detection_loop(lanes):
    print("[INFO] Starting detection thread...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam index", WEBCAM_INDEX)
        return

    # Load YOLO model once
    model = YOLO(WEIGHTS)
    # initialize deques
    for ln in lanes.keys():
        _current_counts[ln] = deque(maxlen=ROLLING_WINDOW)

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed")
            time.sleep(0.5)
            continue

        # run YOLO inference (small resize for speed)
        h, w = frame.shape[:2]
        # optional: resize to 640 width for speed
        target_w = 640
        scale = target_w / w if w > target_w else 1.0
        if scale != 1.0:
            proc = cv2.resize(frame, (int(w*scale), int(h*scale)))
        else:
            proc = frame

        results = model(proc, conf=0.4, iou=0.45)

        # count vehicles per lane for this frame
        counts_this_frame = {ln: 0 for ln in lanes.keys()}

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in ["car", "bus", "truck", "motorcycle"]:
                    # scale box back if resized
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if scale != 1.0:
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    for ln, poly in lanes.items():
                        if point_in_polygon((cx, cy), poly):
                            counts_this_frame[ln] += 1
                            break

        # update rolling history
        with _lock:
            for ln, v in counts_this_frame.items():
                if ln not in _current_counts:
                    _current_counts[ln] = deque(maxlen=ROLLING_WINDOW)
                _current_counts[ln].append(v)

            # compute averages
            avg = {}
            for ln, dq in _current_counts.items():
                avg[ln] = float(sum(dq)) / (len(dq) if len(dq) else 1)

            # choose signal
            lane, duration = choose_signal_from_counts(avg)
            _current_signal["lane"] = lane
            _current_signal["duration"] = duration
            _current_signal["counts"] = avg

        # show debug window with annotated frame (optional)
        # draw lanes and counts:
        vis = frame.copy()
        for ln, poly in lanes.items():
            cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 200, 0), 2)
            pos = tuple(poly.mean(axis=0).astype(int))
            text = f"{ln}: {counts_this_frame.get(ln, 0)}"
            cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

        if _current_signal["lane"]:
            cv2.putText(vis, f"GREEN -> {_current_signal['lane']} ({_current_signal['duration']}s)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Traffic - Press q to quit", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # maintain FPS
        elapsed = time.time() - start
        to_sleep = max(0, 1.0 / FPS - elapsed)
        time.sleep(to_sleep)

    cap.release()
    cv2.destroyAllWindows()


# ---------- Flask API ----------
app = Flask("traffic_api")


@app.route("/signal", methods=["GET"])
def get_signal():
    with _lock:
        resp = {
            "lane": _current_signal.get("lane"),
            "duration": _current_signal.get("duration"),
            "counts": _current_signal.get("counts")
        }
    return jsonify(resp)


def run_flask():
    app.run(host="0.0.0.0", port=API_PORT, debug=False)


# ---------- Main ----------
if __name__ == "__main__":
    # load lanes (expects lanes_config.json prepared)
    lanes = load_lanes()
    if not lanes:
        print(
            "[ERROR] No lanes found. Run calibration first and save lanes_config.json.")
        exit(1)

    # start detection thread
    t = threading.Thread(target=detection_loop, args=(lanes,), daemon=True)
    t.start()

    # start flask (main thread)
    print(f"[INFO] Flask API serving on port {API_PORT}. Endpoint: /signal")
    run_flask()
