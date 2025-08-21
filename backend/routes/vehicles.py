from fastapi import APIRouter, UploadFile, File
from database import db, serialize_doc
from models import Vehicle
from datetime import datetime
import cv2
import numpy as np
from preprocessing import preprocess_pipeline as preprocess_frame

vehicle_router = APIRouter()


@vehicle_router.post("/vehicles")
async def create_vehicle(vehicle: Vehicle):
    vehicle_dict = vehicle.dict()
    vehicle_dict["timestamp"] = datetime.utcnow().isoformat()
    result = await db.vehicles.insert_one(vehicle_dict)
    new_vehicle = await db.vehicles.find_one({"_id": result.inserted_id})
    return serialize_doc(new_vehicle)


@vehicle_router.get("/vehicles")
async def get_vehicles():
    vehicles = []
    cursor = db.vehicles.find({})
    async for doc in cursor:
        vehicles.append(serialize_doc(doc))
    return vehicles


@vehicle_router.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    """
    Upload an image, preprocess it with OpenCV, and return processed result.
    """
    contents = await file.read()

    # Convert to numpy array for OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Use your preprocessing function
    processed = preprocess_frame(frame)

    # Encode processed image back to base64 or jpg bytes
    _, buffer = cv2.imencode(".jpg", processed)
    return {"processed_frame": buffer.tobytes().hex()}
