#!/usr/bin/env python3
"""
auto_lane_yolo.py

- Manual lane calibration per image (mouse)
- Global lanes_config.json stores lanes for many images
- Re-calibration overwrites lanes for that image in the same JSON file
- Detection mode loads lanes by image filename and runs YOLO (yolov8n.pt) for vehicle counting
"""

import cv2
import json
import os
import time
import numpy as np
from ultralytics import YOLO

LANES_FILE = "lanes_config.json"

# ---------------- Utility: load/save lanes ----------------


def load_all_lanes(config_path=LANES_FILE):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        try:
            data = json.load(f)
            return data
        except Exception:
            return {}


def save_all_lanes(all_lanes, image_key, config_path=LANES_FILE):
    # Save lanes under the correct image key
    with open(config_path, "w") as f:
        json.dump(all_lanes, f, indent=4)

    print(
        f"✅ Saved {len(all_lanes[image_key])} polygons for image '{image_key}'.")


def image_key_from_path(image_path):
    return os.path.basename(image_path)

# ---------------- Calibration (manual drawing) ----------------


current_polygon = []
all_polygons_for_image = {}  # name -> list of (x,y)
scale = 1.0
_auto_lane_counter = 1


def mouse_callback(event, x, y, flags, param):
    global current_polygon, all_polygons_for_image, scale, _auto_lane_counter
    orig_x = int(x / scale)
    orig_y = int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((orig_x, orig_y))
        print(f"Point added: {(orig_x, orig_y)}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        # finish polygon automatically and auto-name it
        if len(current_polygon) >= 3:
            lane_name = f"Lane_{_auto_lane_counter}"
            _auto_lane_counter += 1
            all_polygons_for_image[lane_name] = current_polygon.copy()
            print(f"Polygon saved for {lane_name}: {current_polygon}")
        else:
            print("Need at least 3 points to form a polygon.")
        current_polygon.clear()


def draw_polygons_on_image(img):
    overlay = cv2.resize(img, None, fx=scale, fy=scale)

    # Draw saved polygons
    for name, polygon in all_polygons_for_image.items():
        pts = (np.array(polygon) * scale).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], True, (0, 200, 0), 2)
        # label near the first point
        p = tuple((np.array(polygon[0]) * scale).astype(int))
        cv2.putText(overlay, name, p, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

    # Draw current polygon (in-progress)
    if len(current_polygon) > 1:
        pts = (np.array(current_polygon) *
               scale).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], False, (255, 100, 0), 2)

    # If single point, draw it
    if len(current_polygon) == 1:
        p = tuple((np.array(current_polygon[0]) * scale).astype(int))
        cv2.circle(overlay, p, 4, (255, 100, 0), -1)

    return overlay


def run_calibration(image_path, config_path=LANES_FILE, display_width=1200):
    global all_polygons_for_image, current_polygon, scale, _auto_lane_counter
    all_polygons_for_image = {}
    current_polygon = []
    _auto_lane_counter = 1

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: cannot read image: {image_path}")
        return

    h, w = img.shape[:2]
    scale = display_width / w if w > display_width else 1.0

    cv2.namedWindow("Lane Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Lane Calibration", mouse_callback)

    print("\nCalibration Instructions:")
    print("  - Left click to add point")
    print("  - Right click to finish polygon (auto-named) and save it for this image")
    print("  - Press 's' to save all polygons for this image and exit (will overwrite existing)")
    print("  - Press 'q' to quit without saving")
    print("  - Press 'r' to reset all drawn polygons for this image")

    # If existing lanes for image exist in global file, preload them to allow re-editing/overwrite
    all_lanes = load_all_lanes(config_path)
    key = image_key_from_path(image_path)
    # print(f"\n🖊️  Calibrating lanes for image: '{key}'")
    if key in all_lanes:
        # ensure polygons exist in right format
        try:
            for name, pts in all_lanes[key].items():
                all_polygons_for_image[name] = [
                    tuple(map(int, p)) for p in pts]
            print(
                f"Loaded existing {len(all_polygons_for_image)} polygons for image '{key}'. They'll be overwritten on save.")
            if all_polygons_for_image:
                _auto_lane_counter = len(all_polygons_for_image) + 1
        except Exception:
            all_polygons_for_image = {}

    while True:
        display = draw_polygons_on_image(img)
        cv2.imshow("Lane Calibration", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q'):
            print("❌ Exiting calibration without saving.")
            break
        elif key == ord('s'):
            # overwrite this image's entry

            image_key = os.path.basename(image_path)

            lanes_json = load_all_lanes(config_path)
            # convert to plain lists
            lanes_json[image_key] = {name: [[int(x), int(y)] for (x, y) in pts]
                                     for name, pts in all_polygons_for_image.items()}

            save_all_lanes(lanes_json, image_key, config_path)
            print(
                f"✅ Saved {len(all_polygons_for_image)} polygons for image '{image_key}'.")
            break
        elif key == ord('r'):
            all_polygons_for_image = {}
            current_polygon = []
            _auto_lane_counter = 1
            print("🔄 Reset all drawn polygons for this image.")
    cv2.destroyAllWindows()

# ---------------- Detection (load lanes and YOLO vehicles) ----------------


def find_lane_for_vehicle(center, lanes):
    # lanes: dict name -> ndarray Nx2
    for name, polygon in lanes.items():
        # cv2.pointPolygonTest expects points in float32 or float64
        if cv2.pointPolygonTest(polygon.astype(np.float32), tuple(center), False) >= 0:
            return name
    return None


def detect_vehicles_yolo(image, lanes, yolo_weights="yolov8n.pt", conf=0.45, iou=0.45):
    # Load model (Ultralytics YOLO)
    model = YOLO(yolo_weights)

    results = model(image, conf=conf, iou=iou)
    vehicle_counts = {lane: {"car": 0, "bus": 0, "truck": 0,
                             "motorbike": 0} for lane in lanes.keys()}
    total_counts = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}

    annotated = image.copy()

    # iterate results and boxes
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in ["car", "bus", "truck", "motorcycle"]:
                if label == "motorcycle":
                    label = "motorbike"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                lane_name = find_lane_for_vehicle(center, lanes)
                if lane_name:
                    vehicle_counts[lane_name][label] += 1
                    total_counts[label] += 1

                    cv2.rectangle(annotated, (x1, y1),
                                  (x2, y2), (200, 30, 30), 2)
                    cv2.putText(annotated, f"{label} [{lane_name}]", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 2)
                else:
                    # draw vehicles not in any lane but still show
                    cv2.rectangle(annotated, (x1, y1),
                                  (x2, y2), (100, 100, 100), 1)
                    cv2.putText(annotated, f"{label}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # draw lanes outlines and labels
    for name, polygon in lanes.items():
        pts = polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], True, (0, 200, 0), 2)
        pos = tuple(polygon.mean(axis=0).astype(int))
        cv2.putText(annotated, name, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

    return annotated, vehicle_counts, total_counts

# ---------------- Helpers ----------------


def resize_for_display(image, width=1200):
    h, w = image.shape[:2]
    if w <= width:
        return image
    scale = width / w
    return cv2.resize(image, (width, int(h * scale)))

# ---------------- Main ----------------


def main():
    print("=== Manual Lane Calibration + YOLO Vehicle Counting ===")
    mode = input(
        "Enter mode (calibration: 'c'  OR  detection: 'd'): ").strip().lower()

    image_path = input("Enter image path (relative or absolute): ").strip()
    if not image_path:
        print("No image path provided. Exiting.")
        return
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    key = image_key_from_path(image_path)

    if mode == "c":
        run_calibration(image_path, LANES_FILE)
    elif mode == "d":
        # load lanes
        all_lanes = load_all_lanes(LANES_FILE)
        if key not in all_lanes:
            print(f"❌ No lanes found for image '{key}' in {LANES_FILE}.")
            print("Run calibration mode first for this image (mode 'c').")
            return

        # convert to numpy arrays for pointPolygonTest and drawing
        lanes = {}
        try:
            for name, pts in all_lanes[key].items():
                arr = np.array(pts, dtype=np.int32)
                lanes[name] = arr
        except Exception as e:
            print("Error parsing lanes for image:", e)
            return

        # read image and run detector
        image = cv2.imread(image_path)
        print("\n🧠 Loading pre-calibrated lanes for image:", key)
        time.sleep(0.7)

        print(
            "🚗 Running YOLO vehicle detection (this may download weights on first run)...")
        annotated, vehicle_counts, total_counts = detect_vehicles_yolo(
            image, lanes)

        # print results
        icons = {"car": "🚗", "bus": "🚌", "truck": "🚚", "motorbike": "🏍️"}
        print("\n🚦 Vehicle counts per lane:")
        for lane, counts in vehicle_counts.items():
            lane_output = f"  {lane}: "
            nonzero = False
            for v_type, v_count in counts.items():
                if v_count > 0:
                    lane_output += f"{icons[v_type]} x {v_count}  "
                    nonzero = True
            if not nonzero:
                lane_output += "No vehicles"
            print(lane_output)

        print("\n📊 Total counts:")
        for v_type, v_count in total_counts.items():
            print(f"  {icons[v_type]} {v_type}: {v_count}")

        # show annotated image
        win = "Final Output - Lane Vehicle Detection"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, resize_for_display(annotated))
        print("\nPress any key on the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Unknown mode. Use 'c' or 'd'.")


if __name__ == "__main__":
    main()
