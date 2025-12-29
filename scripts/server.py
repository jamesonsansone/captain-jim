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
from openai import OpenAI
from fastapi.responses import Response

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CaptainJimServer")

load_dotenv()

ai_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 STARTUP: Initializing AI Systems...")
    try:
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("❌ ERROR: OPENAI_API_KEY is missing.")
        
        embedding_function = OpenAIEmbeddings()
        
        # Ensure this points to where you saved your DB
        db_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        
        db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        
        # --- UPDATED RETRIEVER SETTINGS ---
        # 1. search_type="mmr": Ensures diversity. It picks the best match, 
        #    then penalizes the next match if it's too similar to the first.
        # 2. k=5: We fetch 5 results now for more detail.
        retriever = db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        ai_resources["retriever"] = retriever
        ai_resources["openai"] = openai_client
        logger.info("✅ AI SYSTEMS READY.")
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
    yield
    ai_resources.clear()

app = FastAPI(lifespan=lifespan)

# --- CORS ---
origins = [
    "https://captain-jim.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class SpeakRequest(BaseModel):
    text: str

def clean_excerpt_text(text):
    """
    Trims text to the last complete sentence to avoid trailing fragments.
    """
    # Find the last occurrence of major punctuation
    last_dot = text.rfind('.')
    last_excl = text.rfind('!')
    last_ques = text.rfind('?')
    
    cutoff = max(last_dot, last_excl, last_ques)
    
    # If punctuation found, trim to it. Otherwise, return as is.
    if cutoff != -1:
        return text[:cutoff+1]
    return text

@app.get("/")
async def health_check():
    return {"status": "online", "message": "Captain Jim Archive is Active"}

@app.post("/ask")
async def ask_captain(request: QueryRequest):
    if "retriever" not in ai_resources:
        raise HTTPException(status_code=503, detail="AI System is not ready yet.")

    try:
        # 1. Retrieve
        docs = ai_resources["retriever"].invoke(request.question)
        
        # 2. FILTERING: Remove "Header Only" or "Empty" chunks
        valid_docs = []
        for d in docs:
            content = d.page_content.strip()
            # If the chunk is just a Chapter header or very short, SKIP IT.
            if len(content) < 100: 
                continue
            valid_docs.append(d)

        # If we filtered everything out, return a polite "No info found"
        if not valid_docs:
            return {
                "summary": "I could not find specific details in the memoir regarding that query.",
                "excerpts": []
            }

        context_text = "\n\n".join([f"Excerpt: {d.page_content}" for d in valid_docs])

        # 3. Historian Prompt (Same as before)
        system_instruction = (
            "You are an expert World War II historian... [KEEP EXISTING PROMPT]"
        )

        completion = ai_resources["openai"].chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {request.question}"}
            ]
        )
        
        summary = completion.choices[0].message.content

        # 4. Format Excerpts
        excerpts_payload = []
        for doc in valid_docs:  # Use the VALID list
            cleaned_text = clean_excerpt_text(doc.page_content)
            excerpts_payload.append({
                "text": cleaned_text,
                "chapter": doc.metadata.get("source", "Memoir Excerpt")
            })

        return {"summary": summary, "excerpts": excerpts_payload}

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speak")
async def generate_audio(request: SpeakRequest):
    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    if not voice_id or not api_key:
        raise HTTPException(status_code=500, detail="Audio configuration missing.")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    
    # --- UPDATED VOICE SETTINGS BASED ON YOUR SCREENSHOT ---
# Inside @app.post("/speak")
    data = {
        "text": request.text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.30,
            "similarity_boost": 0.95,
            "style": 0.20,
            "use_speaker_boost": True,
            "speed": 0.8  
        }
    }
    
    response = requests.post(url, json=data, headers=headers)
    return Response(content=response.content, media_type="audio/mpeg")