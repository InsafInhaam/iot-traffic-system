import cv2
import numpy as np

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
    lane_width = w // 4
    lanes = [(i * lane_width, (i+1) * lane_width) for i in range(4)]
    return lanes

# Function to check which lane a vehicle center belongs to


def find_lane_for_vehicle(center, lanes):
    x, _ = center
    for i, (start, end) in enumerate(lanes):
        if start <= x < end:
            return i + 1  # lane numbering starts from 1
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


# ---------------- Execution ----------------
if __name__ == "__main__":
    image_path = "intersection.png"  # replace with your intersection image
    image, gray, binary, blurred, edges = preprocess_image(image_path)

    lanes = detect_lanes(image)

    vehicles, vehicle_counts, centers = detect_vehicles(image)

    # Draw lanes
    for start, end in lanes:
        cv2.line(image, (start, 0), (start, image.shape[0]), (0, 255, 0), 2)

    # Draw detected vehicles and assign lane
    for (x, y, w, h), center in zip(vehicles, centers):
        lane_number = find_lane_for_vehicle(center, lanes)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, f"Lane {lane_number}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    print("Vehicle counts:", vehicle_counts)
    print("Justification:", justification)

    # Show outputs
    cv2.imshow("Original", image)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Binary", binary)
    cv2.imshow("Blurred", blurred)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
