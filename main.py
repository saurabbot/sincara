from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_community.document_loaders.csv_loader import CSVLoader
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from pydantic import BaseModel
from langchain.chains.question_answering import load_qa_chain
import uuid, os, openai, tiktoken
from langchain.vectorstores.pinecone import Pinecone as pinecone

from pinecone import Pinecone, PodSpec, Config


from typing import List, Optional
from schemas.company import GetCompanySchema
from utils.pincone_utils import create_new_pinecone_index

app = FastAPI()

client = AsyncIOMotorClient(os.getenv("MONGO_URL"))
database = client["convoai"]
collection = database["companies"]


class Company(BaseModel):
    _id: str
    name: str
    domain: str


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/companies/", response_model=Company)
async def create_company(company: Company):
    company_dict = {
        "name": company.name,
        "domain": company.domain,
        "pinecone_index": f"pc-{company.name}",
        "uuid": str(uuid.uuid4()),
    }
    result = await collection.insert_one(company_dict)
    return {**company.dict(), "_id": str(result.inserted_id)}


@app.get("/companies/", response_model=List[GetCompanySchema])
async def read_companies():
    companies = []
    companies_collection = collection.find({})
    async for company in companies_collection:
        print(company)
        company_dict = company.copy()
        company_dict["_id"] = str(company_dict.pop("_id"))
        companies.append(GetCompanySchema(**company_dict))
    return companies


@app.get("/companies/{company_uuid}")
async def create_pinecone_index_for_company(company_uuid: str):
    company = await collection.find_one({"uuid": company_uuid})
    if company:
        message = create_new_pinecone_index(company)
        return {"message": message}
    raise HTTPException(status_code=404, detail="Company not found")


openai.api_key = os.getenv("OPENAI_API_KEY")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
SMART_CHAT_MODEL = "gpt-4"
FAST_CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"


@app.post("/companies/{company_uuid}/upload/")
async def upload_csv(company_uuid: str, file: UploadFile = File(...)):
    company = await collection.find_one({"uuid": company_uuid})
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = f"uploads/{file.filename}"
    if os.path.exists(file_path):
        print("File already exists")
    else:
        with open(file_path, "wb") as f:
            f.write(file.file.read())
    loader = CSVLoader(file_path=file_path, encoding="utf-8")
    data = loader.load()
    print(data)
    print(f"You have loaded a PDF with {len(data)} pages")
    print(f"There are {len(data[0].page_content)} characters in your document")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(data)
    print(f"You have split your document into {len(texts)} smaller documents")
    print("Creating embeddings and index...")
    embeddings = OpenAIEmbeddings(client="")
    for i, doc in enumerate(texts):
        doc.metadata["source"] = f"document_{i+1}"
    docsearch = PineconeVectorStore.from_texts(
        [t.page_content for t in texts],
        embeddings,
        index_name=company["pinecone_index"],
        metadatas=[t.metadata for t in texts],
    )
    print("Done!")
    return {"message": "Upload done!"}


@app.post("/companies/{company_uuid}/query/")
async def query_context(company_uuid: str, query: str):
    company = await collection.find_one({"uuid": company_uuid})
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    print("Querying...")
    print(company["pinecone_index"])

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"), model=EMBEDDING_MODEL
    )
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
    )
    index = pc.Index(company["pinecone_index"])
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="pc-repurpose", embedding=embeddings
    )

    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name=FAST_CHAT_MODEL),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    result = qa({"question": query})

    return {"result": result}

