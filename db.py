import os
from motor.motor_asyncio import AsyncIOMotorClient

client = AsyncIOMotorClient(os.getenv("MONGO_URL"))
database = client["convoai"]
