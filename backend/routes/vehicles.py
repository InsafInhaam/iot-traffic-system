from fastapi import APIRouter
from database import db, serialize_doc
from models import Vehicle
from datetime import datetime

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
