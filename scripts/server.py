import os
import requests
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi.responses import Response

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CaptainJimServer")

load_dotenv()

# Global variables to hold our AI "Brain"
ai_resources = {}

# --- LIFESPAN MANAGER (The Fix) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the AI
    logger.info("🚀 STARTUP: Initializing AI Systems...")
    try:
        embedding_function = OpenAIEmbeddings()
        # Ensure we are looking in the right place for the DB
        db_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        
        db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        retriever = db.as_retriever(search_kwargs={"k": 4})
        
        # Store in global state
        ai_resources["retriever"] = retriever
        ai_resources["llm"] = llm
        
        logger.info("✅ AI SYSTEMS READY. Server is listening.")
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR during startup: {e}")
    
    yield
    
    # Shutdown: Clean up resources if needed
    logger.info("🛑 SHUTDOWN: Server stopping...")
    ai_resources.clear()

# Initialize App with the Lifespan
app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
class QueryRequest(BaseModel):
    question: str

class SpeakRequest(BaseModel):
    text: str

# --- ENDPOINTS ---

@app.get("/")
async def health_check():
    """Simple ping to prove the server is alive."""
    if not ai_resources:
        return {"status": "loading", "message": "AI is still waking up..."}
    return {"status": "online", "message": "Captain Jim Archive is Active"}

@app.post("/ask")
async def ask_captain(request: QueryRequest):
    if "retriever" not in ai_resources:
        raise HTTPException(status_code=503, detail="AI System is not ready yet.")

    logger.info(f"Received question: {request.question}")
    
    try:
        # 1. Retrieve
        docs = ai_resources["retriever"].invoke(request.question)
        
        # 2. Summarize
        summary_prompt = ChatPromptTemplate.from_template(
            "You are a military historian summarizing events from Captain James V. Morgia's memoir. "
            "Answer the question strictly based on the context provided below. "
            "Write in the third person. Do NOT use 'I'. Be detailed but factual.\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION: {input}"
        )
        chain = create_stuff_documents_chain(ai_resources["llm"], summary_prompt)
        summary = chain.invoke({"context": docs, "input": request.question})

        # 3. Format
        excerpts_payload = []
        for doc in docs[:3]: 
            excerpts_payload.append({
                "text": doc.page_content,
                "chapter": doc.metadata.get("source", "Unknown Chapter")
            })

        return {"summary": summary, "excerpts": excerpts_payload}

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speak")
async def generate_audio(request: SpeakRequest):
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not voice_id or not api_key:
        raise HTTPException(status_code=500, detail="ElevenLabs API Keys missing.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": request.text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ElevenLabs Error: {response.text}")
        return Response(content=response.content, media_type="audio/mpeg")
    except Exception as e:
        logger.error(f"Audio failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))