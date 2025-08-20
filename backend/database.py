from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

MONGO_URL = "mongodb+srv://insafinhaam732:zI3f9OSXjbkGjHUa@cluster0.cng0d5o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "traffic_system"

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

# Helper to convert Mongo ObjectId → str
def serialize_doc(doc):
    return {**doc, "_id": str(doc["_id"])}
