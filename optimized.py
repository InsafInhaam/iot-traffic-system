"""
Simple end-to-end traffic camera -> tracker -> adaptive controller
One-file script. Features:
 - Calibration UI to draw lanes and (optionally) set homography (4 points)
 - Lightweight centroid tracker (persistent IDs)
 - YOLOv8 detection (ultralytics) for vehicles
 - Simple adaptive green-time allocation based on queue length (meters)
 - Non-blocking control sender (HTTP) with retries

Usage:
 1) Install dependencies: pip install ultralytics opencv-python numpy requests
 2) Run: python traffic_controller.py
 3) Follow prompts: 'c' to capture calibration image and draw lanes, or 'w' to run live detection

Save lanes/homography to lanes_config.json in working dir.
"""

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import cv2
import json
import os
import time
import numpy as np
import threading
import queue
import requests
from ultralytics import YOLO

# --------- CONFIG (edit these) ----------
LANES_FILE = "lanes_config.json"
DEFAULT_IMAGE_SAVE = "calib_frame.jpg"
YOLO_WEIGHTS = "yolov8n.pt"
CONF = 0.45
IOU = 0.45
VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "motorbike", "truck"}
EMERGENCY_LABELS = {"ambulance", "fire engine", "firetruck", "police"}
NODEMCU_IP = "192.168.1.7"  # change if needed
BASE_CONTROL_URL = f"http://{NODEMCU_IP}/control"
BASE_MODE_URL = f"http://{NODEMCU_IP}/mode"
MIN_GREEN_MS = 3000
MAX_GREEN_MS = 10000
YELLOW_MS = 2000
EMERGENCY_GREEN_MS = 8000
CONTROL_INTERVAL_MS = 500
COUNT_WINDOW_SECS = 2.0
DEFAULT_VEHICLE_LENGTH_M = 4.5

# --------- utils for saving/loading ---------


def load_all_lanes(path=LANES_FILE):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def save_all_lanes(j, path=LANES_FILE):
    with open(path, 'w') as f:
        json.dump(j, f, indent=2)

# ---------- simple centroid tracker ---------


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_id = 1
        self.objects = {}  # id -> centroid
        self.disappeared = {}  # id -> frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, rects):
        # rects: list of (x1,y1,x2,y2)
        if len(rects) == 0:
            # increment disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        input_centroids = []
        for (x1, y1, x2, y2) in rects:
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for c in input_centroids:
                self._register(c)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = np.linalg.norm(np.array(object_centroids)[
                               :, None, :] - np.array(input_centroids)[None, :, :], axis=2)
            # match greedily
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > self.max_distance:
                    continue
                oid = object_ids[r]
                self.objects[oid] = input_centroids[c]
                self.disappeared[oid] = 0
                used_rows.add(r)
                used_cols.add(c)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            # mark disappeared
            for r in unused_rows:
                oid = object_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

            # register new objects
            for c in unused_cols:
                self._register(input_centroids[c])
        return self.objects

    def _register(self, centroid):
        oid = self.next_id
        self.next_id += 1
        self.objects[oid] = centroid
        self.disappeared[oid] = 0

    def _deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
        if oid in self.disappeared:
            del self.disappeared[oid]


# ---------- mouse / calibration UI -------------
current_polygon = []
all_polygons_for_image = {}
_auto_lane_counter = 1
scale = 1.0

homography_src = []  # image points
homography_dst = []  # meter points
H = None


def mouse_callback(event, x, y, flags, param):
    global current_polygon, all_polygons_for_image, _auto_lane_counter, scale, homography_src
    orig_x = int(x / scale)
    orig_y = int(y / scale)
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((orig_x, orig_y))
        print("Point added:", (orig_x, orig_y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_polygon) >= 3:
            lane_name = f"Lane_{_auto_lane_counter}"
            _auto_lane_counter += 1
            all_polygons_for_image[lane_name] = current_polygon.copy()
            print(f"Polygon saved for {lane_name}")
        else:
            print("Need >=3 points to form polygon")
        current_polygon = []


def draw_polygons_on_image(img):
    overlay = cv2.resize(img, None, fx=scale, fy=scale)
    for name, polygon in all_polygons_for_image.items():
        pts = (np.array(polygon) * scale).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], True, (0, 200, 0), 2)
        pos = tuple((np.array(polygon[0]) * scale).astype(int))
        cv2.putText(overlay, name, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
    if len(current_polygon) > 1:
        pts = (np.array(current_polygon) *
               scale).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], False, (255, 100, 0), 2)
    if len(current_polygon) == 1:
        p = tuple((np.array(current_polygon[0]) * scale).astype(int))
        cv2.circle(overlay, p, 4, (255, 100, 0), -1)
    return overlay


def run_calibration_from_image(image_path, display_width=1200):
    global all_polygons_for_image, current_polygon, scale, _auto_lane_counter, homography_src, homography_dst, H
    all_polygons_for_image = {}
    current_polygon = []
    _auto_lane_counter = 1

    img = cv2.imread(image_path)
    if img is None:
        print("Cannot read:", image_path)
        return
    h, w = img.shape[:2]
    scale = display_width / w if w > display_width else 1.0

    cv2.namedWindow("Lane Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Lane Calibration", mouse_callback)
    print("Left-click add points. Right-click finish polygon. 's' save, 'h' homography, 'r' reset, 'q' quit")

    # preload
    all_lanes = load_all_lanes()
    key = os.path.basename(image_path)
    if key in all_lanes:
        try:
            for name, pts in all_lanes[key].items():
                all_polygons_for_image[name] = [
                    tuple(map(int, p)) for p in pts]
            _auto_lane_counter = len(all_polygons_for_image) + 1
            print(f"Loaded {len(all_polygons_for_image)} polygons for {key}")
        except Exception:
            all_polygons_for_image = {}

    while True:
        display = draw_polygons_on_image(img)
        cv2.imshow("Lane Calibration", display)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            print("Exit without saving")
            break
        if k == ord('s'):
            lanes_json = load_all_lanes()
            lanes_json[key] = {name: [[int(x), int(y)] for (
                x, y) in pts] for name, pts in all_polygons_for_image.items()}
            save_all_lanes(lanes_json)
            print(f"Saved {len(all_polygons_for_image)} polygons for {key}")
            break
        if k == ord('r'):
            all_polygons_for_image = {}
            current_polygon = []
            _auto_lane_counter = 1
            print("Reset polygons")
        if k == ord('h'):
            # start homography capture - ask user to click 4 points in order
            homography_src = []
            homography_dst = []
            print(
                "Click 4 image points (top-left -> clockwise). Press 'd' when done to enter real coords.")
            while len(homography_src) < 4:
                img2 = display.copy()
                cv2.putText(img2, f"Click image pt {len(homography_src)+1}/4",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Lane Calibration", img2)
                k2 = cv2.waitKey(1) & 0xFF
                if k2 == ord('q'):
                    break
                # capture clicks via mouse handler storing into current_polygon temporarily
                # reuse left click to add to homography_src
                # we'll poll current_polygon
                if len(current_polygon) > 0:
                    homography_src.append(current_polygon[0])
                    current_polygon = []
                    print("Captured image point", homography_src[-1])
            if len(homography_src) == 4:
                print(
                    "Enter real-world coordinates for each image point in meters as 'x y' (e.g. 0 0")
                for i in range(4):
                    s = input(f"Point {i+1} meters x y: ")
                    try:
                        xm, ym = map(float, s.strip().split())
                        homography_dst.append([xm, ym])
                    except Exception:
                        print("Bad input, aborting homography")
                        homography_src = []
                        homography_dst = []
                        break
                if len(homography_dst) == 4:
                    H, _ = cv2.findHomography(
                        np.array(homography_src), np.array(homography_dst))
                    cfg = load_all_lanes()
                    key = os.path.basename(image_path)
                    if key not in cfg:
                        cfg[key] = {}
                    cfg[key]['_homography_src'] = homography_src
                    cfg[key]['_homography_dst'] = homography_dst
                    save_all_lanes(cfg)
                    print("Homography saved to lanes config")
    cv2.destroyAllWindows()

# ------------- geometric helpers ----------------


def pixel_to_world(pt):
    # pt = (x,y), return (xm, ym) if H exists else None
    global H
    if H is None:
        return None
    p = np.array([pt[0], pt[1], 1.0])
    w = H.dot(p)
    if abs(w[2]) < 1e-6:
        return None
    w = w / w[2]
    return (float(w[0]), float(w[1]))


# ------------ networking (non-blocking) --------------
control_q = queue.Queue()
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.3)
session.mount('http://', HTTPAdapter(max_retries=retries))


def control_worker():
    while True:
        lane, color, state = control_q.get()
        try:
            params = {"lane": lane, "color": color, "state": int(state)}
            try:
                r = session.get(BASE_CONTROL_URL, params=params, timeout=1.5)
                print(
                    f"Sent control -> lane:{lane} color:{color} state:{state} | {r.status_code}")
            except Exception as e:
                print("Control send failed:", e)
        finally:
            control_q.task_done()


threading.Thread(target=control_worker, daemon=True).start()


def set_node_mode(mode):
    try:
        r = session.get(BASE_MODE_URL, params={"set": mode}, timeout=1.2)
        print("Mode set response:", r.status_code)
    except Exception as e:
        print("Error setting mode:", e)

# ------------- Adaptive controller (simple) ----------------


class AdaptiveController:
    def __init__(self, lanes, min_green_ms=MIN_GREEN_MS, max_green_ms=MAX_GREEN_MS):
        self.lanes = list(lanes)
        self.min_green_ms = min_green_ms
        self.max_green_ms = max_green_ms
        self.current_green = None
        self.green_end = 0

    def next_action(self, aggregated_counts, emergency_lane=None):
        now = int(time.time() * 1000)
        if emergency_lane:
            if self.current_green != emergency_lane:
                print("EMERGENCY ->", emergency_lane)
                set_node_mode('manual')
                control_q.put((emergency_lane, 'green', 1))
                for l in self.lanes:
                    if l != emergency_lane:
                        control_q.put((l, 'red', 1))
                self.current_green = emergency_lane
                self.green_end = now + EMERGENCY_GREEN_MS
            return
        if self.current_green and now < self.green_end:
            return
        # choose best lane by queue meters
        best, best_val = None, 0.0
        for l, val in aggregated_counts.items():
            if val > best_val:
                best_val = val
                best = l
        if best is None or best_val <= 0.0:
            # all red fallback
            for l in self.lanes:
                control_q.put((l, 'red', 1))
            self.current_green = None
            self.green_end = now + 1000
            return
        # map meters to green time using saturation (vehicles per second)
        saturation_vps = 0.7  # vehicles per second per lane (conservative)
        estimated_veh = max(1.0, best_val / DEFAULT_VEHICLE_LENGTH_M)
        required_secs = estimated_veh / saturation_vps
        green_ms = int(np.clip(required_secs * 1000,
                       self.min_green_ms, self.max_green_ms))
        print(
            f"Choose {best}: queue_m={best_val:.1f}m est_veh={estimated_veh:.1f} green_ms={green_ms}")
        set_node_mode('manual')
        control_q.put((best, 'green', 1))
        for l in self.lanes:
            if l != best:
                control_q.put((l, 'red', 1))
        self.current_green = best
        self.green_end = now + green_ms

# ------------- main detection loop ----------------


def find_lane_for_point(pt, lanes_np):
    for name, poly in lanes_np.items():
        if cv2.pointPolygonTest(poly.astype(np.float32), tuple(pt), False) >= 0:
            return name
    return None


def run_webcam_detection(device_index=0):
    all_lanes = load_all_lanes()
    if not all_lanes:
        print("No lanes found. Run calibration first.")
        return
    print("Available keys:", list(all_lanes.keys()))
    if len(all_lanes) == 1:
        key = list(all_lanes.keys())[0]
    else:
        key = input("Enter lane image key to load (filename): ").strip()
        if key not in all_lanes:
            print("Key not found")
            return
    cfg = all_lanes[key]
    # load homography if present
    global H
    if '_homography_src' in cfg and '_homography_dst' in cfg:
        try:
            H, _ = cv2.findHomography(
                np.array(cfg['_homography_src']), np.array(cfg['_homography_dst']))
            print("Homography loaded")
        except Exception:
            H = None
    lanes_np = {}
    for name, pts in cfg.items():
        if name.startswith('_'):
            continue
        lanes_np[name] = np.array([[int(p[0]), int(p[1])]
                                  for p in pts], dtype=np.int32)
    lane_names = list(lanes_np.keys())
    if len(lane_names) == 0:
        print("No polygon lanes found in config")
        return

    controller = AdaptiveController(lane_names)
    print("Loading model...")
    model = YOLO(YOLO_WEIGHTS)

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    tracker = CentroidTracker()
    track_history = []  # (ts, track_id, lane, label)
    window_ms = int(COUNT_WINDOW_SECS * 1000)
    last_control = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No frame")
                break
            results = model(frame, conf=CONF, iou=IOU, verbose=False)
            annotated = frame.copy()
            rects = []
            labels = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label == 'motorcycle':
                        label = 'motorbike'
                    if label not in VEHICLE_LABELS and label not in EMERGENCY_LABELS:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    rects.append((x1, y1, x2, y2))
                    labels.append(label)
                    cv2.rectangle(annotated, (x1, y1),
                                  (x2, y2), (200, 30, 30), 2)
                    cv2.putText(annotated, label, (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 0), 2)

            objects = tracker.update(rects)
            # map centroids back to rect index by nearest centroid
            rect_centroids = [((r[0]+r[2])//2, (r[1]+r[3])//2) for r in rects]
            # build mapping from object id to associated label & bbox
            oid_to_label = {}
            for oid, centroid in objects.items():
                # find nearest rect centroid
                if len(rect_centroids) == 0:
                    continue
                dists = [np.hypot(centroid[0]-c[0], centroid[1]-c[1])
                         for c in rect_centroids]
                idx = int(np.argmin(dists))
                if idx < len(labels):
                    oid_to_label[oid] = labels[idx]
                # draw
                cv2.putText(annotated, f"ID:{oid}", (
                    centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(annotated, centroid, 4, (0, 255, 0), -1)

            # maintain history
            now_ms = int(time.time()*1000)
            for oid, centroid in objects.items():
                label = oid_to_label.get(oid, 'car')
                lane = find_lane_for_point(centroid, lanes_np)
                if lane:
                    track_history.append((now_ms, oid, lane, label, centroid))

            # prune
            track_history = [
                t for t in track_history if now_ms - t[0] <= window_ms]

            # aggregated counts: compute queue length in meters per lane
            aggregated_m = {l: 0.0 for l in lane_names}
            # simple approach: unique track ids per lane * vehicle length, optionally refine with homography
            seen = {l: set() for l in lane_names}
            for ts, oid, lane, label, centroid in track_history:
                if oid in seen[lane]:
                    continue
                seen[lane].add(oid)
                # if homography available, try world coords
                w = pixel_to_world(centroid)
                if w is not None:
                    # approximate lane direction by projecting centroid into lane centerline is omitted for simplicity
                    # use vehicle length estimate
                    aggregated_m[lane] += DEFAULT_VEHICLE_LENGTH_M
                else:
                    aggregated_m[lane] += DEFAULT_VEHICLE_LENGTH_M

            # draw lane info
            for name, poly in lanes_np.items():
                pts = poly.reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], True, (0, 200, 0), 2)
                pos = tuple(poly.mean(axis=0).astype(int))
                cv2.putText(annotated, f"{name} {aggregated_m.get(name, 0):.0f}m",
                            pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # emergency detection
            emergency_lane = None
            for ts, oid, lane, label, centroid in track_history:
                if label in EMERGENCY_LABELS:
                    emergency_lane = lane
                    break

            if now_ms - last_control >= CONTROL_INTERVAL_MS:
                controller.next_action(
                    aggregated_m, emergency_lane=emergency_lane)
                last_control = now_ms

            display = cv2.resize(annotated, (1280, int(
                annotated.shape[0]*1280/annotated.shape[1])))
            cv2.imshow('Live Detection', display)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('p'):
                cv2.imwrite('snapshot.jpg', frame)
                print('snapshot saved')
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ------------- main CLI ----------------


def main():
    print("=== Simple Traffic Controller ===")
    print("Modes: 'c' calibration (capture frame), 'w' webcam detection")
    mode = input("Enter mode ('c' or 'w'): ").strip().lower()
    if mode == 'c':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Cannot open camera')
            return
        print("Press 'c' to capture frame and start drawing lanes. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print('frame fail')
                break
            cv2.imshow('Calibration Capture - press c', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('c'):
                cv2.imwrite(DEFAULT_IMAGE_SAVE, frame)
                print('Saved frame to', DEFAULT_IMAGE_SAVE)
                cap.release()
                cv2.destroyAllWindows()
                run_calibration_from_image(DEFAULT_IMAGE_SAVE)
                break
            if k == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
    elif mode == 'w':
        run_webcam_detection(0)
    else:
        print('Unknown mode')


if __name__ == '__main__':
    main()
