from pydantic import BaseModel


class GetCompanySchema(BaseModel):
    name: str
    domain: str
    pinecone_index: str
    _id: str
    uuid: str
