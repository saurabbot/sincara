from pydantic import BaseModel


class Customer(BaseModel):
    name: str
    phone: str
    company: str
    _id: str
    uuid: str
    token: str
    created_at: str
    updated_at: str
    deleted_at: str
    is_active: bool


class BrandUser(BaseModel):
    name: str
    email: str
    company: str
    _id: str
    uuid: str
    is_brand_admin: bool
    created_at: str
    updated_at: str
