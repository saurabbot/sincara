from schemas.company import GetCompanySchema
from pinecone import Pinecone, PodSpec
import os


def create_new_pinecone_index(company: GetCompanySchema):
    company_pinecone_index = company["pinecone_index"]
    print(f"Creating new Pinecone index for company: {company_pinecone_index}")
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
    )
    indexes = pc.list_indexes().indexes
    existing_index_names = [index["name"] for index in indexes]
    if company_pinecone_index in existing_index_names:
        print(f"Index {company_pinecone_index} already exists")
        return "Index already exists"
    else:
        pc.create_index(
            name=company_pinecone_index,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter", pod_type="s1.x1"),
        )
        print(f"Index {company_pinecone_index} created successfully")
        return "Index created successfully"
