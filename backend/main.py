from fastapi import FastAPI
from routes.vehicles import vehicle_router

app = FastAPI()

app.include_router(vehicle_router, prefix="/vehicles")
