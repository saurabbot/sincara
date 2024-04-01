from fastapi import APIRouter, HTTPException, UploadFile, File, Request
import os, openai, tiktoken
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, Config
from langchain.chains import RetrievalQAWithSourcesChain
from twilio.rest import Client
from langchain.chat_models.openai import ChatOpenAI
from db import database

router = APIRouter()
collection = database["companies"]

openai.api_key = os.getenv("OPENAI_API_KEY")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
SMART_CHAT_MODEL = "gpt-4"
FAST_CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"


@router.post("/{company_uuid}/upload/")
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
        index_name="convo-ai",  # company["pinecone_index"],
        metadatas=[t.metadata for t in texts],
    )
    print("Done!")
    return {"message": "Upload done!"}


@router.post("/{company_uuid}/query/")
async def query_context(company_uuid: str, request: Request):
    company = await collection.find_one({"uuid": company_uuid})
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    data = await request.form()
    incoming_number = data.get("From", "").split(":")[1]  # e.g., "+1234567890"
    print(f"Received message from {incoming_number}")
    query = data.get("Body", "").lower()
    print("Querying...")
    print(company["pinecone_index"])

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"), model=EMBEDDING_MODEL
    )
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
    )
    index = pc.Index("convo-ai")  # company["pinecone_index"])
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="convo-ai", embedding=embeddings  # company["pinecone_index"]
    )

    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name=FAST_CHAT_MODEL),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    result = qa({"question": query})
    if result is not None:
        account_sid = os.getenv("TWILLIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILLIO_AUTH_TOKEN")
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            to=f"whatsapp:{incoming_number}",
            from_="whatsapp:+14155238886",
            body=result["answer"],
        )
        print(message.sid)
    return str(result["answer"])
