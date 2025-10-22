import cv2
import json
import numpy as np

# Global variables
drawing = False
current_polygon = []
all_polygons = {}


def mouse_callback(event, x, y, flags, param):
    global drawing, current_polygon

    if event == cv2.EVENT_LBUTTONDOWN:
        # Left-click → add a point
        current_polygon.append((x, y))
        print(f"Point added: {(x, y)}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right-click → finish current polygon
        if len(current_polygon) >= 3:
            lane_name = input("Enter lane name (e.g., L1, L2, Side_Left): ")
            all_polygons[lane_name] = current_polygon.copy()
            print(f"Polygon saved for {lane_name}: {current_polygon}")
        current_polygon = []

    elif event == cv2.EVENT_MBUTTONDOWN:
        # Middle-click → save all polygons to JSON and exit
        with open("lanes_config.json", "w") as f:
            json.dump(all_polygons, f, indent=4)
        print("✅ All polygons saved to lanes_config.json")
        cv2.destroyAllWindows()


def draw_polygons(img):
    overlay = img.copy()
    # Draw all saved polygons
    for name, polygon in all_polygons.items():
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
        cv2.putText(overlay, name, tuple(polygon[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw current polygon in progress
    if len(current_polygon) > 1:
        pts = np.array(current_polygon, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], False, (255, 0, 0), 2)

    return overlay


if __name__ == "__main__":
    image_path = "samples/image2.avif"  # change to your intersection snapshot
    img = cv2.imread(image_path)
    cv2.namedWindow("Lane Calibration")
    cv2.setMouseCallback("Lane Calibration", mouse_callback)

    print("Instructions:")
    print("🖱️ Left click = add point")
    print("🖱️ Right click = finish polygon and name lane")
    print("🖱️ Middle click = save all polygons and exit")

    while True:
        display = draw_polygons(img)
        cv2.imshow("Lane Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC = quit without saving
            break
        elif key == ord('s'):  # "s" = save + exit
            with open("lanes_config.json", "w") as f:
                json.dump(all_polygons, f, indent=4)
            print("✅ All polygons saved to lanes_config.json")
            break

    cv2.destroyAllWindows()