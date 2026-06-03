from dotenv import load_dotenv
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agents.graph import create_graph
import chat

    
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    chat.graph = await create_graph()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "OPTIONS", "DELETE"],
    allow_headers=["Content-Type"],
)

@app.get("/")
def index():
    return "Ready"

@app.post("/api/chat")
async def chat_endpoint(message: chat.ChatMessage):
    return await chat.chat(message)

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)

if __name__ == "__main__":
    main()