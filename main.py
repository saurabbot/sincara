from fastapi import FastAPI
from pyngrok import ngrok
from routes.company import router as company_router
from routes.data import router as data_router
import sys

app = FastAPI()


port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8000"
public_url = ngrok.connect(port).public_url
print(f'ngrok tunnel "{public_url}" -> "http://127.0.0.1:{port}")')


@app.get("/")
async def read_root():
    return {"Hello": "World"}


app.include_router(company_router, prefix="/companies", tags=["companies"])
app.include_router(data_router, prefix="/data", tags=["data"])
