# import cv2
# from ultralytics import YOLO

# # Load pretrained YOLOv3 model
# model = YOLO('yolov3.pt')

# vehicle_classes = {"car", "truck", "bus", "motorbike"}

# def detect_vehicles(image):
#     results = model(image)[0]  # get first batch (image) results

#     vehicle_count = 0

#     for box, cls, score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
#         class_id = int(cls.item())
#         label = model.names[class_id]
#         if label in vehicle_classes and score > 0.5:
#             vehicle_count += 1
#             x1, y1, x2, y2 = map(int, box)
#             color = (0, 255, 0)  # green for vehicles
#             cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(image, f"{label} {score:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     cv2.putText(image, f"Vehicles: {vehicle_count}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

#     return image, vehicle_count

# # Video processing
# # cap = cv2.VideoCapture("samples/video.mp4")
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     annotated_frame, count = detect_vehicles(frame)

#     print(f"Detected vehicles: {count}")
#     cv2.imshow("Vehicle Detection", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
