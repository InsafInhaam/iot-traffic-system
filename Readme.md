# 🚦 IoT Traffic Optimization System (Vehicle Detection Prototype)

This is a prototype for counting and classifying vehicles in an image using YOLOv3 and OpenCV. It is the first step in building an intelligent traffic control system using computer vision.

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/InsafInhaam/iot-traffic-system.git
cd iot-traffic-system
```

---

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

---

### 3️⃣ Install Required Python Libraries

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Create the `model` Folder & Download YOLOv3 Files

Since model files are **not pushed to GitHub** (too large), create the folder and download them manually:

```bash
mkdir model
cd model
```

#### 🔽 Download YOLOv3 ONNX model

```bash
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov3/model/yolov3-8.onnx
```

#### 📄 coco.names

```bash
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

Once downloaded, go back to the main project folder:

```bash
cd ..
```

---

### 5️⃣ Run the Detection Script

```bash
python detect.py
```

Press **Q** to quit the live detection window.

python3 preprocessing_technique.py

python3 preprocessing_technique_v2.py
