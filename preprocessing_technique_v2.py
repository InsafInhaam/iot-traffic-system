import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- Task 1: Preprocessing ----------------


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return image, gray, binary, blurred, edges


# ---------------- Task 2: Lane Separator Logic ----------------
def detect_lanes(image):
    h, w, _ = image.shape

    # Example: each main lane is split into left/right halves
    lanes = {
        "L1_left": np.array([(0, 0), (w//8, 0), (w//8, h), (0, h)], np.int32),
        "L1_right": np.array([(w//8, 0), (w//4, 0), (w//4, h), (w//8, h)], np.int32),

        "L2_left": np.array([(w//4, 0), (3*w//8, 0), (3*w//8, h), (w//4, h)], np.int32),
        "L2_right": np.array([(3*w//8, 0), (w//2, 0), (w//2, h), (3*w//8, h)], np.int32),

        "L3_left": np.array([(w//2, 0), (5*w//8, 0), (5*w//8, h), (w//2, h)], np.int32),
        "L3_right": np.array([(5*w//8, 0), (3*w//4, 0), (3*w//4, h), (5*w//8, h)], np.int32),

        "L4_left": np.array([(3*w//4, 0), (7*w//8, 0), (7*w//8, h), (3*w//4, h)], np.int32),
        "L4_right": np.array([(7*w//8, 0), (w, 0), (w, h), (7*w//8, h)], np.int32),

        # Example side road
        "side_left": np.array([(0, h//2), (w//6, h//2), (w//6, h), (0, h)], np.int32),
        "side_right": np.array([(5*w//6, h//2), (w, h//2), (w, h), (5*w//6, h)], np.int32)
    }
    return lanes


def find_lane_for_vehicle(center, lanes):
    for name, polygon in lanes.items():
        if cv2.pointPolygonTest(polygon, center, False) >= 0:
            return name
    return None


# ---------------- Task 3: Justification ----------------
justification = {
    "🌑 Grayscale": "Reduces computation by simplifying 🎨 3 color channels into one intensity channel.",
    "⚫ Binary": "Helps in distinguishing vehicles from the background by highlighting contrasts ⬛⬜.",
    "🌫️ Gaussian Blur": "Removes noise before edge detection 🔎.",
    "✂️ Canny Edges": "Highlights lane lines and vehicle edges 🛣️.",
    "🛣️ Lane Division": "Splitting into lanes (↔️ left/right + side roads) allows accurate vehicle classification 🚗🚌🚚🏍️.",
}

# ---------------- Task 4: Vehicle Detection with YOLO ----------------


def detect_vehicles_yolo(image, lanes):
    model = YOLO('yolov8n.pt')

    results = model(image)

    # Initialize lane counts dynamically
    vehicle_counts = {lane: {"car": 0, "bus": 0, "truck": 0,
                             "motorbike": 0} for lane in lanes.keys()}
    total_counts = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}

    annotated = image.copy()

    # Detect and classify
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["car", "bus", "truck", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2)//2, (y1 + y2)//2)

                lane_name = find_lane_for_vehicle(center, lanes)
                if lane_name:
                    vehicle_counts[lane_name][label] += 1
                    total_counts[label] += 1

                    # Draw bounding box with icon-style label
                    cv2.rectangle(annotated, (x1, y1),
                                  (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated, f"{label} [{lane_name}]", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw lane polygons
    for name, polygon in lanes.items():
        cv2.polylines(annotated, [polygon], isClosed=True,
                      color=(0, 255, 0), thickness=2)
        # Label lanes
        # text_pos = tuple(polygon.mean(axis=0).astype(int)[0])
        text_pos = tuple(polygon.mean(axis=0).astype(int))
        cv2.putText(annotated, name, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Overlay total vehicle counts on top-left
    y_offset = 30
    for v_type, count in total_counts.items():
        cv2.putText(annotated, f"{v_type}: {count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    return annotated, vehicle_counts, total_counts


# ---------------- Utility ----------------
def resize_for_display(image, width=960):
    h, w = image.shape[:2]
    scale = width / w
    return cv2.resize(image, (width, int(h * scale)))


# ---------------- Execution ----------------
if __name__ == "__main__":
    image_path = "samples/image2.avif"
    image, gray, binary, blurred, edges = preprocess_image(image_path)

    lanes = detect_lanes(image)

    annotated, vehicle_counts, total_counts = detect_vehicles_yolo(
        image, lanes)

    # Icons for each vehicle type
    icons = {
        "car": "🚗",
        "bus": "🚌",
        "truck": "🚚",
        "motorbike": "🏍️"
    }

    print("\n" + "-"*50 + "\n")
    print("🚦 Vehicle counts per lane:")
    for lane, counts in vehicle_counts.items():
        lane_output = f"  {lane}: "
        for v_type, v_count in counts.items():
            if v_count > 0:  # only show if present
                lane_output += f"{icons[v_type]} x {v_count}  "
        if lane_output.strip() == f"{lane}:":
            lane_output += "No vehicles"
        print(lane_output)

    print("\n" + "-"*50 + "\n")

    print("📊 Total counts:")
    for v_type, v_count in total_counts.items():
        print(f"  {icons[v_type]} {v_type}: {v_count}")

    print("\n" + "-"*50 + "\n")
    print("📌 Justifications:")
    for step, reason in justification.items():
        print(f"{step}: {reason}")

    print("\n" + "-"*50 + "\n")

    # Show outputs resized for any screen
    cv2.imshow("YOLO Detection", resize_for_display(annotated))
    cv2.imshow("Original", resize_for_display(image))
    cv2.imshow("Grayscale", resize_for_display(gray))
    cv2.imshow("Binary", resize_for_display(binary))
    cv2.imshow("Blurred", resize_for_display(blurred))
    cv2.imshow("Edges", resize_for_display(edges))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
