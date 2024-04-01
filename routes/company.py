from fastapi import APIRouter, HTTPException
from utils.pincone_utils import create_new_pinecone_index
from schemas.company import Company, GetCompanySchema
import uuid
from typing import List, Optional
from db import database

router = APIRouter()
collection = database["companies"]


@router.post("/", response_model=Company)
async def create_company(company: Company):
    company_dict = {
        "name": company.name,
        "domain": company.domain,
        "pinecone_index": f"pc-{company.name}",
        "uuid": str(uuid.uuid4()),
    }
    result = await collection.insert_one(company_dict)
    return {**company.dict(), "_id": str(result.inserted_id)}


@router.get("/", response_model=List[GetCompanySchema])
async def read_companies():
    companies = []
    companies_collection = collection.find({})
    async for company in companies_collection:
        print(company)
        company_dict = company.copy()
        company_dict["_id"] = str(company_dict.pop("_id"))
        companies.append(GetCompanySchema(**company_dict))
    return companies


@router.get("/{company_uuid}")
async def create_pinecone_index_for_company(company_uuid: str):
    company = await collection.find_one({"uuid": company_uuid})
    if company:
        # message = create_new_pinecone_index(company) forbidden since we are on a free plan of Pinecone
        return {"message": company}
    raise HTTPException(status_code=404, detail="Company not found")
