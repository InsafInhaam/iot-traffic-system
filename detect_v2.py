import cv2
from ultralytics import YOLO

# Load pretrained YOLOv3 model
model = YOLO('yolov3.pt')

vehicle_classes = {"car", "truck", "bus", "motorbike"}
emergency_keywords = {"ambulance", "fire truck",
                      "police car"}  # depends on model training

# Define lanes (adjust coordinates for your camera view)
# Format: (x_start, y_start, x_end, y_end)
lanes = {
    "Lane 1": (0, 0, 300, 720),
    "Lane 2": (300, 0, 600, 720),
    "Lane 3": (600, 0, 900, 720)
}


def detect_vehicles(image):
    results = model(image)[0]  # first batch result

    lane_counts = {lane: 0 for lane in lanes}
    total_vehicles = 0
    emergency_detected = False

    for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        class_id = int(cls.item())
        label = model.names[class_id]

        if label in vehicle_classes and score > 0.5:
            total_vehicles += 1
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Lane assignment
            for lane_name, (lx1, ly1, lx2, ly2) in lanes.items():
                if lx1 <= cx <= lx2 and ly1 <= cy <= ly2:
                    lane_counts[lane_name] += 1

            # Emergency vehicle keyword check
            if any(keyword in label.lower() for keyword in emergency_keywords):
                emergency_detected = True

            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Emergency vehicle siren color check (basic)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = (0, 120, 70)
    upper_red = (10, 255, 255)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    if cv2.countNonZero(mask_red) > 800:  # adjust threshold
        emergency_detected = True

    # Display total vehicle count
    cv2.putText(image, f"Total Vehicles: {total_vehicles}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Display lane-wise counts
    y_offset = 60
    for lane_name, count in lane_counts.items():
        cv2.putText(image, f"{lane_name}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 30

    # Emergency vehicle alert
    if emergency_detected:
        cv2.putText(image, "EMERGENCY VEHICLE DETECTED", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    return image, total_vehicles, lane_counts, emergency_detected


# Video capture
# cap = cv2.VideoCapture(0)  # webcam
cap = cv2.VideoCapture("samples/video6.mp4")  # video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame, total_count, lane_counts, emergency = detect_vehicles(
        frame)

    print(
        f"Total Vehicles: {total_count}, Lanes: {lane_counts}, Emergency: {emergency}")
    cv2.imshow("Vehicle & Lane Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
