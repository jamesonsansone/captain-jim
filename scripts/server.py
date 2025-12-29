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
        db_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        retriever = db.as_retriever(search_kwargs={"k": 4})
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        ai_resources["retriever"] = retriever
        ai_resources["openai"] = openai_client
        logger.info("✅ AI SYSTEMS READY.")
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
    yield
    ai_resources.clear()

app = FastAPI(lifespan=lifespan)

# --- SECURITY: CORS CONFIGURATION ---
origins = [
    "https://captain-jim.vercel.app",  # Your specific Vercel App
    "http://localhost:5500",           # For local testing
    "http://127.0.0.1:5500"            # For local testing
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
        context_text = "\n\n".join([f"Excerpt: {d.page_content}" for d in docs])

        # 2. Historian Prompt (UPDATED)
        system_instruction = (
            "You are an expert World War II historian. You are receiving a question from a user who believes they are speaking directly to the spirit or legacy of Captain James V. Morgia. "
            "Your task is to answer their question using the specific details from his memoir, 'Three Day Pass'.\n\n"
            "Style Guide:\n"
            "1. **Referencing Jim:** In the first sentence, refer to him as 'Captain James V. Morgia'. In all subsequent sentences, refer to him warmly as 'Captain Jim'.\n"
            "2. **Tone:** Authoritative, respectful, and narrative. Use the provided context to tell a story.\n"
            "3. **Perspective:** Write in the third person (he/him/Jim), but answer the user's question directly.\n"
            "4. **Accuracy:** Stick strictly to the facts provided in the context."
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

        # 3. Format Excerpts
        excerpts_payload = []
        for doc in docs[:3]: 
            excerpts_payload.append({
                "text": doc.page_content,
                "chapter": doc.metadata.get("source", "Unknown Chapter")
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
    data = {
        "text": request.text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
