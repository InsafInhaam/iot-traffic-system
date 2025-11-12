import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import time
from sklearn.cluster import DBSCAN

# ---------------- Preprocessing ----------------

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return image, gray, binary, blurred, edges


# ---------------- Lane Detection (Deep Learning) ----------------

def detect_lanes_deep_learning(image, model_path="yolov8n-seg.pt"):
    """
    Detect lanes automatically using a segmentation model.
    You can use a YOLOv8 segmentation model trained for lanes (e.g., yolov8n-lane.pt).
    """
    print("🧠 Running deep learning lane detection...")

    model = YOLO(model_path)
    results = model(image)

    lanes = {}
    annotated = image.copy()
    lane_id = 1

    for result in results:
        if hasattr(result, "masks") and result.masks is not None:
            for mask in result.masks.xy:
                lane_points = np.array(mask, dtype=np.int32)
                lane_name = f"Lane_{lane_id}"
                lanes[lane_name] = lane_points
                lane_id += 1

                # Draw the detected lane
                cv2.polylines(annotated, [lane_points], False, (0, 255, 0), 2)
                cv2.putText(
                    annotated, lane_name, tuple(lane_points[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

    if not lanes:
        print("⚠️ No lanes detected — try a better model or image.")
    else:
        print(f"✅ Detected {len(lanes)} lanes automatically.")

    return annotated, lanes


def detect_lanes_deep_learning_v2(image,
                                  model_path="yolov8n-seg.pt",
                                  min_area=1500,
                                  min_length=80,
                                  clustering_eps=50,
                                  clustering_min_samples=1,
                                  debug=False):
    """
    Improved lane detection with postprocessing:
      - filters small mask components
      - clusters nearby mask fragments (DBSCAN)
      - unions cluster masks and fits a smooth polyline per lane

    Args:
      image: BGR image (numpy array)
      model_path: segmentation model path (YOLOv8 seg)
      min_area: minimum connected-component area to keep (px)
      min_length: minimum polyline length (px) to be considered a lane
      clustering_eps: DBSCAN epsilon (pixels) for merging fragments
      clustering_min_samples: DBSCAN min samples
      debug: if True, returns debug overlay and prints extra info

    Returns:
      annotated (image), lanes (dict: name -> Nx2 int32 poly points)
    """
    # load model & run
    model = YOLO(model_path)
    results = model(image)

    h, w = image.shape[:2]

    # Collect mask components
    # list of dicts: {'mask': mask_uint8, 'area': area, 'centroid': (x,y)}
    comp_masks = []
    for result in results:
        if not hasattr(result, "masks") or result.masks is None:
            continue
        # result.masks.xy is a list of polygons; but also we can get binary mask via result.masks.data or result.masks.masks
        # We will use result.masks.data if available; otherwise rasterize polygons.
        # Try to get binary mask array if present
        try:
            # Ultralytics provides result.masks.data or result.masks.masks as a (n, h, w) boolean array in many versions
            if hasattr(result.masks, "data") and result.masks.data is not None:
                # result.masks.data might be (n, h, w) boolean
                mask_array = result.masks.data  # boolean array (n, h, w)
                for i in range(mask_array.shape[0]):
                    mask = (mask_array[i].astype(np.uint8) * 255)
                    # connected components to separate fragments
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        mask, connectivity=8)
                    for lbl in range(1, num_labels):
                        area = int(stats[lbl, cv2.CC_STAT_AREA])
                        if area < min_area:
                            continue
                        comp_mask = (labels == lbl).astype(np.uint8) * 255
                        cy, cx = int(centroids[lbl, 1]), int(centroids[lbl, 0])
                        comp_masks.append(
                            {'mask': comp_mask, 'area': area, 'centroid': (cx, cy)})
            else:
                # fallback: rasterize polygons from result.masks.xy
                for poly in result.masks.xy:
                    poly_pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly_pts], 255)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        mask, connectivity=8)
                    for lbl in range(1, num_labels):
                        area = int(stats[lbl, cv2.CC_STAT_AREA])
                        if area < min_area:
                            continue
                        comp_mask = (labels == lbl).astype(np.uint8) * 255
                        cy, cx = int(centroids[lbl, 1]), int(centroids[lbl, 0])
                        comp_masks.append(
                            {'mask': comp_mask, 'area': area, 'centroid': (cx, cy)})
        except Exception as e:
            # best-effort fallback to polygon list
            for poly in getattr(result.masks, "xy", []):
                poly_pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly_pts], 255)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    mask, connectivity=8)
                for lbl in range(1, num_labels):
                    area = int(stats[lbl, cv2.CC_STAT_AREA])
                    if area < min_area:
                        continue
                    comp_mask = (labels == lbl).astype(np.uint8) * 255
                    cy, cx = int(centroids[lbl, 1]), int(centroids[lbl, 0])
                    comp_masks.append(
                        {'mask': comp_mask, 'area': area, 'centroid': (cx, cy)})

    if len(comp_masks) == 0:
        if debug:
            print("No mask components after area filtering")
        return image.copy(), {}

    # Build centroids array for clustering
    centroids = np.array([c['centroid'] for c in comp_masks], dtype=np.float32)

    # DBSCAN to cluster nearby fragments into a single lane
    clustering = DBSCAN(eps=clustering_eps, min_samples=clustering_min_samples)
    labels = clustering.fit_predict(centroids)
    unique_labels = [lab for lab in np.unique(labels) if lab != -1]

    lanes = {}
    annotated = image.copy()
    lane_idx = 1

    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        # union masks for this cluster
        union_mask = np.zeros((h, w), dtype=np.uint8)
        for i in idxs:
            union_mask = cv2.bitwise_or(union_mask, comp_masks[i]['mask'])

        # morphological closing to join near fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        union_mask = cv2.morphologyEx(
            union_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # find largest contour
        contours, _ = cv2.findContours(
            union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        # keep the largest by area
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        # optional: approximate contour to polyline and filter by length
        # compute bounding rect height (or contour length)
        length = cv2.arcLength(c, False)
        if length < min_length:
            # skip tiny / short blobs
            continue

        # fit a smooth polyline: sample contour points and fit a polynomial in x vs y
        pts = c.reshape(-1, 2)
        # sort by y (top to bottom)
        pts = pts[np.argsort(pts[:, 1])]
        # reduce to unique y-sampled points to build polyline
        ys = np.linspace(pts[:, 1].min(), pts[:, 1].max(), num=100)
        # For each y, find mean x among points with nearby y
        poly_pts = []
        for yv in ys:
            mask_y = np.abs(pts[:, 1] - yv) < 6  # window
            if np.any(mask_y):
                mean_x = int(np.mean(pts[mask_y][:, 0]))
                poly_pts.append((mean_x, int(yv)))
        if len(poly_pts) < 2:
            continue
        poly_pts = np.array(poly_pts, dtype=np.int32)

        # Simplify polyline
        epsilon = max(3.0, 0.01 * cv2.arcLength(poly_pts, False))
        approx = cv2.approxPolyDP(poly_pts, epsilon, False)
        if approx.shape[0] < 2:
            continue

        lane_name = f"Lane_{lane_idx}"
        lanes[lane_name] = approx.reshape(-1, 2).astype(np.int32)
        lane_idx += 1

        # draw for visualization
        cv2.polylines(annotated, [lanes[lane_name]], False, (0, 255, 0), 2)
        # label near top point
        top_pt = lanes[lane_name][0].tolist()
        cv2.putText(annotated, lane_name, tuple(top_pt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if debug:
        print(
            f"Components kept: {len(comp_masks)}, clusters found: {len(unique_labels)}, lanes returned: {len(lanes)}")
    return annotated, lanes

# ---------------- Vehicle Detection ----------------


def find_lane_for_vehicle(center, lanes):
    for name, polygon in lanes.items():
        if cv2.pointPolygonTest(polygon, center, False) >= 0:
            return name
    return None


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
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                lane_name = find_lane_for_vehicle(center, lanes)
                if lane_name:
                    vehicle_counts[lane_name][label] += 1
                    total_counts[label] += 1

                    cv2.rectangle(annotated, (x1, y1),
                                  (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        annotated, f"{label} [{lane_name}]",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2
                    )

    # Draw detected lanes
    for name, polygon in lanes.items():
        cv2.polylines(annotated, [polygon.astype(
            np.int32)], True, (0, 255, 0), 2)
        text_pos = tuple(polygon.mean(axis=0).astype(int))
        cv2.putText(
            annotated, name, text_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )

    return annotated, vehicle_counts, total_counts


# ---------------- Utility ----------------

def resize_for_display(image, width=1500):
    h, w = image.shape[:2]
    scale = width / w
    return cv2.resize(image, (width, int(h * scale)))


# ---------------- Main ----------------

if __name__ == "__main__":
    image_path = "samples/image2.avif"
    image, gray, binary, blurred, edges = preprocess_image(image_path)

    print("\n🧭 Step 1: Detecting lanes automatically...")
    # annotated_lanes, lanes = detect_lanes_deep_learning(image)
    annotated_lanes, lanes = detect_lanes_deep_learning_v2(
        image,
        model_path="yolov8n-seg.pt",
        min_area=1500,
        min_length=80,
        clustering_eps=50,
        debug=True
    )

    if not lanes:
        print("❌ No lanes detected. Exiting.")
        exit()

    print("\n🚗 Step 2: Detecting vehicles and mapping to lanes...")
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

    cv2.imshow("Final Output - Auto Lane + Vehicle Detection",
               resize_for_display(annotated))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
