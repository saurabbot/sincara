from pydantic import BaseModel, Field, EmailStr
from bson import ObjectId
from ObjectId import PyObjectId


class CustomerModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    phone_number: str = Field(...)
