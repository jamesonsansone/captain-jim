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
        # 1. Retrieve (Now getting 5 diverse results)
        docs = ai_resources["retriever"].invoke(request.question)
        
        # Prepare context for GPT
        context_text = "\n\n".join([f"Excerpt: {d.page_content}" for d in docs])

        # 2. Historian Prompt
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
        for doc in docs: 
            # Clean the text (remove partial sentences at end)
            cleaned_text = clean_excerpt_text(doc.page_content)
            
