from fastapi import APIRouter, HTTPException, UploadFile, File, Request
import os, openai, tiktoken
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, Config
from langchain.chains import RetrievalQAWithSourcesChain
from twilio.rest import Client
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
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
            "customer_id": incoming_number,
            "sent_at": {"$gte": datetime.utcnow() - timedelta(minutes=10)},
        }
    )
    messages_list = []
    prompt = f"""
    Your name is Suresh, a friendly and knowledgeable chatbot assistant for Backstreet Academy, an online clothing company that sells apparel and fashion items through their website.

    Background information about Backstreet Academy:
    - Backstreet Academy is an e-commerce platform specializing in clothing and fashion accessories.
    - The company offers a wide range of products including t-shirts, shirts, dresses, jeans, shoes, and more.
    - Backstreet Academy curates trendy and stylish collections for men, women, and children.
    - The website provides detailed product descriptions, sizing guides, and customer reviews to assist with purchasing decisions.
    - Backstreet Academy also offers periodic sales, discounts, and promotional offers.

    Instructions:
    - Greet the customer warmly and introduce yourself as Suresh, the chatbot assistant for Backstreet Academy.
    - If the customer greets you or initiates a conversation, respond with a friendly greeting and ask how you can assist them.
    - Carefully read and understand the customer's query or message related to clothing, fashion, or the company's products and services.
    - Provide clear, concise, and informative responses, tailored to the customer's specific needs or questions.
    - If the customer inquires about specific products, provide relevant details such as product descriptions, available sizes, colors, pricing, and customer reviews.
    - If the customer has questions about orders, shipping, returns, or other customer service-related queries, assist them to the best of your knowledge.
    - If the customer's query is outside the scope of Backstreet Academy's offerings, politely inform them that you cannot assist with that particular topic.
    - Maintain a friendly, professional, and helpful tone throughout the conversation.
    - If you need additional information from the customer to better understand or respond to their query, ask clarifying questions.
    - Never say I don't know. If you're unsure about a response, offer to find the information or escalate the query to my manager whose name is John and number is +919972502038.
    - if the customer asks for a discount, offer a 10% discount on their next purchase using the code "WELCOME10".
    - if the customer wants to buy a product, ask for their shipping address and contact number to process the order.

    Previous conversation:
    """
    messages_list.append(prompt)
    async for message in messages:
        if message.get("sender_id") == incoming_number:
            messages_list.append(f"Customer: {message.get('message')}")
        else:
            messages_list.append(f"Suresh: {message.get('message')}")
    full_query = "\n".join(messages_list) + f"\nCustomer: {query}"
    print(full_query)
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"), model=EMBEDDING_MODEL
    )
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
    )
    index = pc.Index("convo-ai")  # company["pinecone_index"])
    doc_search = PineconeVectorStore(
        index, embedding=embeddings  # company["pinecone_index"]
    )
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name=FAST_CHAT_MODEL),
        chain_type="stuff",
        retriever=doc_search.as_retriever(
            search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50}
        ),
    )

    result = qa({"question": full_query})
    new_message = {
        "sender_id": incoming_number,
        "receiver_id": company_uuid,
        "customer_id": incoming_number,
        "message": query,
        "sent_at": datetime.utcnow(),
    }
    await message_collection.insert_one(new_message)
    new_sender_message = {
        "sender_id": company_uuid,
        "receiver_id": incoming_number,
        "customer_id": incoming_number,
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
