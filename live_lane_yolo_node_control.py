import cv2
import json
import os
import time
import numpy as np
import threading
import requests
from ultralytics import YOLO

# ---------------- CONFIG ----------------
LANES_FILE = "lanes_config.json"
DEFAULT_IMAGE_SAVE = "calib_frame.jpg"

# NodeMCU / traffic controller base URL (change to your device IP)
NODEMCU_IP = "192.168.1.7"
BASE_CONTROL_URL = f"http://{NODEMCU_IP}/control"
BASE_MODE_URL = f"http://{NODEMCU_IP}/mode"

# YOLO parameters
YOLO_WEIGHTS = "yolov8n.pt"   # change to custom model if you have one
CONF = 0.45
IOU = 0.45

# Vehicle labels we care about (COCO): car, bus, truck, motorcycle
VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle"}

# Labels to treat as EMERGENCY — **adjust** this to your model or dataset
# If your model doesn't include ambulance/fire-engine, you must fine-tune or map classes here.
EMERGENCY_LABELS = {"ambulance", "fire engine",
                    "firetruck", "police"}  # probably empty for vanilla COCO

# Adaptive timing (ms)
MIN_GREEN_MS = 3000
MAX_GREEN_MS = 10000
YELLOW_MS = 2000

# How long to keep emergency green (ms)
EMERGENCY_GREEN_MS = 8000

# How often to send control updates (ms)
# don't spam controller; we check control decisions at this interval
CONTROL_INTERVAL_MS = 500

# Smoothing window for counts (seconds)
COUNT_WINDOW_SECS = 2.0

# ---------------- helper functions ----------------


def load_all_lanes(config_path=LANES_FILE):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def save_all_lanes(all_lanes, config_path=LANES_FILE):
    with open(config_path, "w") as f:
        json.dump(all_lanes, f, indent=2)


def image_key_from_path(path):
    return os.path.basename(path)


# ---------------- Calibration UI (mouse) ----------------
current_polygon = []
all_polygons_for_image = {}
_auto_lane_counter = 1
scale = 1.0


def mouse_callback(event, x, y, flags, param):
    global current_polygon, all_polygons_for_image, _auto_lane_counter, scale
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
            print(
                f"Polygon saved for {lane_name}: {all_polygons_for_image[lane_name]}")
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
    global all_polygons_for_image, current_polygon, scale, _auto_lane_counter
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
    print("Left-click to add points. Right-click to finish polygon (auto-name). 's' save, 'r' reset, 'q' quit without saving")

    # preload if exists
    all_lanes = load_all_lanes()
    key = image_key_from_path(image_path)
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
    cv2.destroyAllWindows()

# ---------------- Detection + control ----------------


def point_in_polygon(pt, polygon):
    return cv2.pointPolygonTest(polygon.astype(np.float32), tuple(pt), False) >= 0


def find_lane_for_center(center, lanes_np):
    for name, poly in lanes_np.items():
        if point_in_polygon(center, poly):
            return name
    return None


def set_node_mode(mode):
    try:
        r = requests.get(BASE_MODE_URL, params={"set": mode}, timeout=1.2)
        print("Mode set response:", r.status_code, r.text)
    except Exception as e:
        print("Error setting mode:", e)


def send_control(lane, color, state):
    try:
        params = {"lane": lane, "color": color, "state": int(state)}
        r = requests.get(BASE_CONTROL_URL, params=params, timeout=1.2)
        print(
            f"Sent control -> lane:{lane} color:{color} state:{state}  | status {r.status_code}")
        return True
    except Exception as e:
        print("Error sending control:", e)
        return False


class AdaptiveController:
    def __init__(self, lanes, min_green_ms=MIN_GREEN_MS, max_green_ms=MAX_GREEN_MS, yellow_ms=YELLOW_MS):
        self.lanes = list(lanes)
        self.min_green_ms = min_green_ms
        self.max_green_ms = max_green_ms
        self.yellow_ms = yellow_ms
        self.current_green_lane = None
        self.green_end_time = 0
        self.cooldowns = {l: 0 for l in self.lanes}

    def next_action(self, aggregated_counts, emergency_lane=None):
        now = int(time.time()*1000)

        # Emergency handling: immediate green for that lane
        if emergency_lane:
            if self.current_green_lane != emergency_lane:
                print("EMERGENCY detected for lane:", emergency_lane)
                # set mode to manual
                set_node_mode("manual")
                # send green to emergency lane and red to opposite
                send_control(emergency_lane, "green", 1)
                # turn all others red
                for l in self.lanes:
                    if l != emergency_lane:
                        send_control(l, "red", 1)
                self.current_green_lane = emergency_lane
                self.green_end_time = now + EMERGENCY_GREEN_MS
            return

        # If currently green and not expired, keep it
        if self.current_green_lane and now < self.green_end_time:
            return

        # green expired or none -> pick lane with highest count (non-zero)
        best_lane = None
        best_count = 0
        for lane, cnt in aggregated_counts.items():
            if cnt > best_count:
                best_count = cnt
                best_lane = lane

        if best_count == 0:
            # no vehicles: fallback to rotating or keep none
            # set all red
            for l in self.lanes:
                send_control(l, "red", 1)
            self.current_green_lane = None
            self.green_end_time = now + 1000  # re-evaluate soon
            return

        # compute green duration proportional to count
        # simple linear mapping between min and max using count (clamped)
        # choose cap_count to map reasonable counts to max green
        cap_count = 10.0
        ratio = min(best_count / cap_count, 1.0)
        green_ms = int(self.min_green_ms + ratio *
                       (self.max_green_ms - self.min_green_ms))

        # set lights: chosen lane green, others red
        print(
            f"Choosing lane {best_lane} with count {best_count}, green_ms={green_ms}")
        set_node_mode("manual")
        send_control(best_lane, "green", 1)
        for l in self.lanes:
            if l != best_lane:
                send_control(l, "red", 1)

        # set yellow timer after green_ms - handled locally here as sleep or scheduling
        self.current_green_lane = best_lane
        self.green_end_time = now + green_ms

# ---------------- Live webcam detector ----------------


def run_webcam_detection(device_index=1):
    # load lanes
    all_lanes = load_all_lanes()
    # choose key: if only one image's lanes exist, use that; else ask user to type key
    if not all_lanes:
        print("No lanes found in lanes_config.json. Run calibration first and save lanes.")
        return

    print("Available image keys in lanes file:", list(all_lanes.keys()))
    # choose key (simple heuristic: if only 1 -> use it)
    if len(all_lanes) == 1:
        image_key = list(all_lanes.keys())[0]
    else:
        image_key = input(
            "Enter lane image key to load (filename from lanes_config.json): ").strip()
        if image_key not in all_lanes:
            print("Key not found. Exiting.")
            return

    # convert to numpy arrays
    lanes_np = {}
    for name, pts in all_lanes[image_key].items():
        # lanes_np[name] = np.array(pts, dtype=np.int32)
        lanes_np[name] = np.array([[int(p[0]), int(p[1])] for p in pts], dtype=np.int32)

    lane_names = list(lanes_np.keys())
    controller = AdaptiveController(lane_names)

    print("Loading model (this may take a moment)...")
    model = YOLO(YOLO_WEIGHTS)

    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # sliding window for counts
    detections_history = []  # list of (timestamp, lane_name, label)
    window_ms = int(COUNT_WINDOW_SECS * 1000)

    last_control_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed")
                break

            results = model(frame, conf=CONF, iou=IOU, verbose=False)

            # reset frame annotation
            annotated = frame.copy()

            # counts for this frame
            frame_counts = {name: {t: 0 for t in [
                "car", "bus", "truck", "motorbike"]} for name in lane_names}
            emergency_lane = None

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label == "motorcycle":
                        label = "motorbike"

                    if label not in VEHICLE_LABELS and label not in EMERGENCY_LABELS:
                        # ignore other labels
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    # cx = (x1 + x2)//2
                    # cy = (y1 + y2)//2
                    cx = (x1 + x2)//2
                    cy = y2 - 5 
                    lane = find_lane_for_center((cx, cy), lanes_np)
                    # draw the box and label
                    cv2.rectangle(annotated, (x1, y1),
                                  (x2, y2), (200, 30, 30), 2)
                    text = label if not lane else f"{label} [{lane}]"
                    cv2.putText(annotated, text, (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 0), 2)

                    if lane:
                        # accumulate detection to history for smoothing
                        ts = int(time.time()*1000)
                        detections_history.append((ts, lane, label))
                        # check emergency
                        if label in EMERGENCY_LABELS:
                            emergency_lane = lane
                        # increment immediate frame counts (only for vehicle labels)
                        if label in ["car", "bus", "truck", "motorbike"]:
                            frame_counts[lane][label] += 1

            # remove old history entries
            now_ms = int(time.time()*1000)
            detections_history = [
                d for d in detections_history if now_ms - d[0] <= window_ms]

            # aggregate counts per lane from history for smoothing
            aggregated_counts = {l: 0 for l in lane_names}
            for ts, lane, lab in detections_history:
                if lab in ["car", "bus", "truck", "motorbike"]:
                    aggregated_counts[lane] += 1

            # draw lanes
            for name, poly in lanes_np.items():
                pts = poly.reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], True, (0, 200, 0), 2)
                pos = tuple(poly.mean(axis=0).astype(int))
                cv2.putText(annotated, f"{name} {aggregated_counts.get(name, 0)}",
                            pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # controller decision every CONTROL_INTERVAL_MS
            if now_ms - last_control_time >= CONTROL_INTERVAL_MS:
                controller.next_action(
                    aggregated_counts, emergency_lane=emergency_lane)
                last_control_time = now_ms

            # show annotated frame
            display = cv2.resize(annotated, (1280, int(
                annotated.shape[0] * 1280/annotated.shape[1])))
            cv2.imshow("Live Detection", display)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('p'):
                # pause and save snapshot
                cv2.imwrite("snapshot.jpg", frame)
                print("Snapshot saved as snapshot.jpg")

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ---------------- Main CLI ----------------


def main():
    print("=== Lane YOLO + NodeMCU controller ===")
    print("Modes: 'c' calibration (capture frame), 'w' webcam detection")
    mode = input("Enter mode ('c' or 'w'): ").strip().lower()

    if mode == 'c':
        # Direct webcam capture for calibration (no more existing image option)
        print("Opening webcam for calibration...")
        print("Press 'c' to capture frame and start drawing lanes.")
        print("Press 'q' to quit.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame failed")
                break

            cv2.imshow("Calibration Capture - press c to capture", frame)
            k = cv2.waitKey(1) & 0xFF

            if k == ord('c'):
                cv2.imwrite(DEFAULT_IMAGE_SAVE, frame)
                print("Saved frame to", DEFAULT_IMAGE_SAVE)

                cap.release()
                cv2.destroyAllWindows()

                # go directly to lane drawing
                run_calibration_from_image(DEFAULT_IMAGE_SAVE)
                break

            if k == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

    elif mode == 'w':
        run_webcam_detection(0)
    else:
        print("Unknown mode")


if __name__ == "__main__":
    main()
