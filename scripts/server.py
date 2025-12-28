import os
import requests
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI # <--- Standard, reliable client
from fastapi.responses import Response

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CaptainJimServer")

load_dotenv()

# Global variables
ai_resources = {}

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 STARTUP: Initializing AI Systems...")
    try:
        # 1. Setup Database Connection (Keep this from LangChain as it works fine)
        embedding_function = OpenAIEmbeddings()
        db_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        retriever = db.as_retriever(search_kwargs={"k": 4})
        
        # 2. Setup Standard OpenAI Client (Replaces broken LangChain logic)
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Store in global state
        ai_resources["retriever"] = retriever
        ai_resources["openai"] = openai_client
        
        logger.info("✅ AI SYSTEMS READY. Server is listening.")
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR during startup: {e}")
    
    yield
    ai_resources.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class SpeakRequest(BaseModel):
    text: str

@app.get("/")
async def health_check():
    return {"status": "online", "message": "Captain Jim Archive is Active"}

@app.post("/ask")
async def ask_captain(request: QueryRequest):
    if "retriever" not in ai_resources:
        raise HTTPException(status_code=503, detail="AI System is not ready yet.")

    try:
        # 1. Retrieve Documents
        docs = ai_resources["retriever"].invoke(request.question)
        
        # 2. Prepare Context Text
        # (We manually join the text chunks, removing the need for LangChain's 'Stuff' chain)
        context_text = "\n\n".join([f"[Excerpt from {d.metadata.get('source', 'Unknown')}]: {d.page_content}" for d in docs])

        # 3. Send to OpenAI Directly
        logger.info("Sending prompt to OpenAI...")
        completion = ai_resources["openai"].chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a military historian summarizing events from Captain James V. Morgia's memoir. Answer the question strictly based on the context provided. Write in the third person. Do NOT use 'I' or 'We'. Be detailed but strictly factual."
                },
                {
                    "role": "user", 
                    "content": f"Context:\n{context_text}\n\nQuestion: {request.question}"
                }
            ]
        )
        
        summary = completion.choices[0].message.content

        # 4. Format Excerpts
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
    # (This code remains exactly the same as before, it works great)
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")

    if not voice_id or not api_key:
        raise HTTPException(status_code=500, detail="ElevenLabs API Keys missing.")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    data = {
        "text": request.text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"ElevenLabs Error: {response.text}")
        
    return Response(content=response.content, media_type="audio/mpeg")