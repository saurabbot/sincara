from fastapi import APIRouter, HTTPException, UploadFile, File, Request
import os, openai, tiktoken
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, Config
from langchain.chains import RetrievalQAWithSourcesChain
from twilio.rest import Client
from datetime import datetime, timedelta
from langchain.chat_models.openai import ChatOpenAI
from db import database

router = APIRouter()

openai.api_key = os.getenv("OPENAI_API_KEY")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
SMART_CHAT_MODEL = "gpt-4"
FAST_CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"


@router.post("/{company_uuid}/upload/")
async def upload_csv(company_uuid: str, file: UploadFile = File(...)):
    collection = database["companies"]
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
    company_collection = database["companies"]
    company = await company_collection.find_one({"uuid": company_uuid})
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    data = await request.form()
    customer_collection = database["customers"]
    message_collection = database["messages"]
    incoming_number = data.get("From", "").split(":")[1]  # e.g., "+1234567890"
    is_existing_customer = await customer_collection.find_one(
        {"phone_number": incoming_number}
    )
    if not is_existing_customer:
        await customer_collection.insert_one({"phone_number": incoming_number})
        print(f"New customer added: {incoming_number}")
    print(f"Received message from {incoming_number}")
    query = data.get("Body", "").lower()
    print("Querying...")
    print(company["pinecone_index"])
    # get all messages of the customer in the past 10 minutes
    messages = message_collection.find(
        {
            "$or": [{"sender_id": incoming_number}, {"sender_id": company_uuid}],
            "sent_at": {"$gte": datetime.utcnow() - timedelta(minutes=10)},
        }
    )
    messages_list = []
    messages_list.append(
        "System: Your a chatbot for a company called backstreet Academy and your name is suresh, Your job is to answer questions from customers related to products and services.If you are unable to answer a question, you should ask the customer for more information."
    )
    async for message in messages:
        if message.get("sender_id") == incoming_number:
            messages_list.append(f"Human: {message.get('message')}")
        else:
            print(message, "AI")
            messages_list.append(f"System: {message.get('message')}")
    full_query = "\n".join(messages_list) + f"\nHuman: {query}"
    print(full_query)
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
    print(docsearch.as_retriever())
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name=FAST_CHAT_MODEL),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    result = qa({"question": full_query})
    new_message = {
        "sender_id": incoming_number,
        "receiver_id": company_uuid,
        "message": query,
        "sent_at": datetime.utcnow(),
    }
    await message_collection.insert_one(new_message)
    new_sender_message = {
        "sender_id": company_uuid,
        "receiver_id": incoming_number,
        "message": result["answer"],
        "sent_at": datetime.utcnow(),
    }
    await message_collection.insert_one(new_sender_message)
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
