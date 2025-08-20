from pydantic import BaseModel
from typing import Optional, Dict


class Vehicle(BaseModel):
    timestamp: str
    lane_counts: Dict[str, int]
    total: int
    emergency: bool
