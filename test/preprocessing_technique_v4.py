import cv2
import numpy as np
from ultralytics import YOLO
import json
import os

# ---------------- Preprocessing ----------------


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return image, gray, binary, blurred, edges

# ---------------- Lane Config Loader ----------------


def load_lanes_from_config(config_path="lanes_config.json"):
    with open(config_path, "r") as f:
        data = json.load(f)
    lanes = {name: np.array(points, np.float32)
             for name, points in data.items()}
    return lanes


def find_lane_for_vehicle(center, lanes):
    for name, polygon in lanes.items():
        if cv2.pointPolygonTest(polygon, center, False) >= 0:
            return name
    return None


# ---------------- Calibration Mode ----------------
drawing = False
current_polygon = []
all_polygons = {}
scale = 1.0  # scale factor for display


def mouse_callback(event, x, y, flags, param):
    global current_polygon, all_polygons, scale

    # Map mouse coordinates to original image
    orig_x = int(x / scale)
    orig_y = int(y / scale)

    if event == cv2.EVENT_LBUTTONDOWN:  # add point
        current_polygon.append((orig_x, orig_y))
        print(f"Point added: {(orig_x, orig_y)}")
    elif event == cv2.EVENT_RBUTTONDOWN:  # finish polygon
        if len(current_polygon) >= 3:
            lane_name = input("Enter lane name (e.g., L1, L2, Side_Left): ")
            all_polygons[lane_name] = current_polygon.copy()
            print(f"Polygon saved for {lane_name}: {current_polygon}")
        current_polygon.clear()


def draw_polygons(img):
    overlay = cv2.resize(img, None, fx=scale, fy=scale)

    # Saved polygons
    for name, polygon in all_polygons.items():
        pts = (np.array(polygon) * scale).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        cv2.putText(overlay, name, tuple((np.array(polygon[0]) * scale).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Current polygon
    if len(current_polygon) > 1:
        pts = (np.array(current_polygon) *
               scale).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], False, (255, 0, 0), 2)

    return overlay


def run_calibration(image_path="samples/image2.avif", config_path="lanes_config.json", display_width=1500):
    global all_polygons, current_polygon, scale
    all_polygons = {}
    current_polygon = []

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    scale = display_width / w

    cv2.namedWindow("Lane Calibration")
    cv2.setMouseCallback("Lane Calibration", mouse_callback)

    print("Calibration Instructions:")
    print("🖱️ Left click = add point")
    print("🖱️ Right click = finish polygon and name lane")
    print("Press 's' to save all polygons and exit")
    print("Press 'q' to quit without saving")

    while True:
        display = draw_polygons(img)
        cv2.imshow("Lane Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("❌ Exiting without saving")
            break
        elif key == ord('s'):
            with open(config_path, "w") as f:
                json.dump(all_polygons, f, indent=4)
            print(f"✅ Lanes saved to {os.path.abspath(config_path)}")
            break

    cv2.destroyAllWindows()

# ---------------- Detection Mode ----------------


def detect_vehicles_yolo(image, lanes):
    model = YOLO("yolov8n.pt")
    results = model(image, conf=0.5, iou=0.5)

    vehicle_counts = {lane: {"car": 0, "bus": 0, "truck": 0,
                             "motorbike": 0} for lane in lanes.keys()}
    total_counts = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}

    annotated = image.copy()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label in ["car", "bus", "truck", "motorcycle"]:
                if label == "motorcycle":
                    label = "motorbike"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1+x2)//2, (y1+y2)//2)

                lane_name = find_lane_for_vehicle(center, lanes)
                if lane_name:
                    vehicle_counts[lane_name][label] += 1
                    total_counts[label] += 1

                    cv2.rectangle(annotated, (x1, y1),
                                  (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated, f"{label} [{lane_name}]", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw polygons
    for name, polygon in lanes.items():
        cv2.polylines(annotated, [polygon.astype(
            np.int32)], True, (0, 255, 0), 2)
        text_pos = tuple(polygon.mean(axis=0).astype(int))
        cv2.putText(annotated, name, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return annotated, vehicle_counts, total_counts

# ---------------- Utility ----------------


def resize_for_display(image, width=1500):
    h, w = image.shape[:2]
    scale = width / w
    return cv2.resize(image, (width, int(h*scale)))


# ---------------- Main ----------------
if __name__ == "__main__":
    mode = input(
        "Enter mode (calibration then click c/detection then click d): ").strip().lower()

    if mode == "c":
        run_calibration("samples/img1.jpg", "lanes_config.json")
    elif mode == "d":
        image_path = "samples/img1.jpg"
        image, gray, binary, blurred, edges = preprocess_image(image_path)
        lanes = load_lanes_from_config("lanes_config.json")
        annotated, vehicle_counts, total_counts = detect_vehicles_yolo(
            image, lanes)

        icons = {"car": "🚗", "bus": "🚌", "truck": "🚚", "motorbike": "🏍️"}

        print("\n🚦 Vehicle counts per lane:")
        for lane, counts in vehicle_counts.items():
            lane_output = f"  {lane}: "
            for v_type, v_count in counts.items():
                if v_count > 0:
                    lane_output += f"{icons[v_type]} x {v_count}  "
            if lane_output.strip() == f"{lane}:":
                lane_output += "No vehicles"
            print(lane_output)

        print("\n📊 Total counts:")
        for v_type, v_count in total_counts.items():
            print(f"  {icons[v_type]} {v_type}: {v_count}")

        # Show outputs
        cv2.imshow("YOLO Detection", resize_for_display(annotated))
        cv2.imshow("Original", resize_for_display(image))
        cv2.imshow("Grayscale", resize_for_display(gray))
        cv2.imshow("Binary", resize_for_display(binary))
        cv2.imshow("Blurred", resize_for_display(blurred))
        cv2.imshow("Edges", resize_for_display(edges))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# ---------------- End ----------------
