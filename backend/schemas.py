from pydantic import BaseModel
from typing import Dict
from datetime import datetime

class VehicleBatch(BaseModel):
    timestamp: datetime
    lane_counts: Dict[str, int]
    total: int
    emergency: bool
