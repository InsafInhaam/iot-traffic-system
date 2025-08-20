from pydantic import BaseModel, Field
from typing import Optional


class Vehicle(BaseModel):
    license_plate: str
    vehicle_type: str
    speed: Optional[float] = None
    timestamp: Optional[str] = None
