from pydantic import BaseModel, Field
from bson import ObjectId


class MessageModel(BaseModel):
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")
    sender_id: ObjectId = Field(...)
    receiver_id: ObjectId = Field(...)
    message: str = Field(...)
    sent_at: str = Field(...)
