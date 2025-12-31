import os
import requests
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request 

# --- LlamaIndex Imports ---
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CaptainJimServer")

load_dotenv()

# Global dictionary to hold loaded AI models
ai_resources = {}

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 STARTUP: Initializing LlamaIndex System...")
    try:
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("❌ ERROR: OPENAI_API_KEY is missing.")

        logger.info("🔹 Loading FastEmbed Model...")
        Settings.embed_model = FastEmbedEmbedding(
            model_name="BAAI/bge-small-en-v1.5", 
            max_length=512
        )
        Settings.llm = None 

        if not os.path.exists(STORAGE_DIR):
             logger.error(f"❌ CRITICAL: Storage directory not found at {STORAGE_DIR}")
        else:
            logger.info(f"🔹 Loading Index from {STORAGE_DIR}...")
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            index = load_index_from_storage(storage_context)
            
            retriever = index.as_retriever(similarity_top_k=5)
            
            ai_resources["retriever"] = retriever
            ai_resources["openai"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            logger.info("✅ CAPTAIN JIM AI READY.")
            
    except Exception as e:
        logger.error(f"❌ CRITICAL STARTUP ERROR: {e}")
    yield
    ai_resources.clear()

# --- APP INITIALIZATION ---
app = FastAPI(lifespan=lifespan)

# --- RATE LIMITER PROXY FIX ---
def get_real_user_ip(request: Request):
    # Render passes the real client IP in the 'x-forwarded-for' header.
    # We take the first IP in the list.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0]
    return "127.0.0.1" # Fallback if no header

# Setup Limiter with the new key function
limiter = Limiter(key_func=get_real_user_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# ------------------------------

# --- ROBUST CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"], # Localhost
    allow_origin_regex=r"https://.*\.vercel\.app", # ANY Vercel Subdomain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class SpeakRequest(BaseModel):
    text: str

def clean_excerpt_text(text):
    last_dot = text.rfind('.')
    last_excl = text.rfind('!')
    last_ques = text.rfind('?')
    cutoff = max(last_dot, last_excl, last_ques)
    if cutoff != -1:
        return text[:cutoff+1]
    return text

@app.get("/")
async def health_check():
    return {"status": "online", "message": "Captain Jim Archive is Active"}

# --- UPDATED ENDPOINT ---
# Note: 'request' must be the system Request object for the limiter to work.
# We renamed the user input to 'query'.
@app.post("/ask")
@limiter.limit("5/minute") 
async def ask_captain(request: Request, query: QueryRequest): 
    if "retriever" not in ai_resources:
        raise HTTPException(status_code=503, detail="AI System is not ready yet.")

    try:
        # We use 'query.question' now instead of 'request.question'
        nodes = ai_resources["retriever"].retrieve(query.question)
        
        valid_nodes = []
        for node in nodes:
            content = node.node.get_content().strip()
            if len(content) < 50: 
                continue
            valid_nodes.append(node)

        if not valid_nodes:
            return {
                "summary": "I searched the archives but couldn't find specific details on that topic in Captain Jim's memoir.",
                "excerpts": []
            }

        context_text = "\n\n".join([f"Excerpt: {n.node.get_content()}" for n in valid_nodes])

        system_instruction = (
            "You are an expert World War 