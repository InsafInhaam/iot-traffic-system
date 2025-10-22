import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- Task 1: Preprocessing (Grayscale + B/W) ----------------


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    _, binary = cv2.threshold(
        gray, 128, 255, cv2.THRESH_BINARY)  # convert to B/W
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return image, gray, binary, blurred, edges

# ---------------- Task 2: Lane Separator Logic ----------------


def detect_lanes(image):
    h, w, _ = image.shape
    # Divide into 4 lanes (simple assumption: equal width vertical lanes)
    # lane_width = w // 4
    # lanes = [(i * lane_width, (i+1) * lane_width) for i in range(4)]
    # return lanes
    lanes = {
        1: np.array([(0, 0), (w//4, 0), (w//4, h), (0, h)], np.int32),
        2: np.array([(w//4, 0), (w//2, 0), (w//2, h), (w//4, h)], np.int32),
        3: np.array([(w//2, 0), (3*w//4, 0), (3*w//4, h), (w//2, h)], np.int32),
        4: np.array([(3*w//4, 0), (w, 0), (w, h), (3*w//4, h)], np.int32),
        # side roads example
        "side_road_left": np.array([(0, h//2), (w//6, h//2), (w//6, h), (0, h)], np.int32)
    }
    return lanes

# Function to check which lane a vehicle center belongs to


def find_lane_for_vehicle(center, lanes):
    # x, _ = center
    # for i, (start, end) in enumerate(lanes):
    #     if start <= x < end:
    #         return i + 1  # lane numbering starts from 1
    # return None
    for name, polygon in lanes.items():
        if cv2.pointPolygonTest(polygon, center, False) >= 0:
            return name
    return None


# ---------------- Task 3: Justification ----------------
justification = {
    "Grayscale": "Reduces computation by simplifying 3 color channels into one intensity channel.",
    "Binary": "Helps in distinguishing vehicles from the background by highlighting contrasts.",
    "Lane Division": "Splitting the frame into equal parts allows classification of vehicles per lane.",
}

# ---------------- Task 4 & 5: Vehicle Detection & Counting ----------------


def detect_vehicles(image):
    # Load pre-trained vehicle classifier (Haar cascade as example)
    # vehicle_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
    vehicle_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vehicles = vehicle_cascade.detectMultiScale(gray, 1.1, 2)

    vehicle_counts = {"car": 0, "van": 0, "others": 0}
    centers = []

    for (x, y, w, h) in vehicles:
        center = (x + w//2, y + h//2)
        centers.append(center)
        if w > 100:  # simple heuristic: larger = van
            vehicle_counts["van"] += 1
        else:
            vehicle_counts["car"] += 1

    return vehicles, vehicle_counts, centers


def detect_vehicles_yolo(image, lanes):
    model = YOLO('yolov8n.pt')  # Load a pre-trained YOLO model

    results = model(image)
    vehicle_counts = {1: {"car": 0, "bus": 0, "truck": 0, "motorbike": 0},
                      2: {"car": 0, "bus": 0, "truck": 0, "motorbike": 0},
                      3: {"car": 0, "bus": 0, "truck": 0, "motorbike": 0},
                      4: {"car": 0, "bus": 0, "truck": 0, "motorbike": 0}}

    annotated = image.copy()

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["car", "bus", "truck", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2)//2, (y1 + y2)//2)

                lane_number = find_lane_for_vehicle(center, lanes)
                if lane_number:
                    vehicle_counts[lane_number][label] += 1

                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1),
                                  (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated, f"{label} L{lane_number}",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
    return annotated, vehicle_counts

# ---------------- Utility Functions ----------------


def resize_for_display(image, width=960):
    h, w = image.shape[:2]
    scale = width / w
    return cv2.resize(image, (width, int(h * scale)))


# ---------------- Execution ----------------
if __name__ == "__main__":
    image_path = "samples/img1.jpg"  # replace with your intersection image
    image, gray, binary, blurred, edges = preprocess_image(image_path)

    lanes = detect_lanes(image)

    # vehicles, vehicle_counts, centers = detect_vehicles(image)

    # Run YOLO
    annotated, vehicle_counts = detect_vehicles_yolo(image, lanes)

    # Draw lanes
    # for start, end in lanes:
    #     cv2.line(image, (start, 0), (start, image.shape[0]), (0, 255, 0), 2)
    for polygon in lanes.values():
        cv2.polylines(annotated, [polygon], isClosed=True,
                      color=(0, 255, 0), thickness=2)

    # Draw detected vehicles and assign lane
    # for (x, y, w, h), center in zip(vehicles, centers):
    #     lane_number = find_lane_for_vehicle(center, lanes)
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #     cv2.putText(image, f"Lane {lane_number}", (x, y-10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    print("Vehicle counts per lane:", vehicle_counts)
    print("Justification:", justification)

    # Show outputs
    cv2.imshow("YOLO Detection", annotated)
    cv2.imshow("Original", resize_for_display(image))
    cv2.imshow("Grayscale", resize_for_display(gray))
    cv2.imshow("Binary", resize_for_display(binary))
    cv2.imshow("Blurred", resize_for_display(blurred))
    cv2.imshow("Edges", resize_for_display(edges))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# test change  
