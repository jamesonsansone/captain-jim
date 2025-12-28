import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# Custom function to split text by "CHAPTER" headings
def load_and_split_by_chapter(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split the text every time we see "CHAPTER" followed by text
    # This creates a list where [0] is the Chapter Title, [1] is the Chapter Content, etc.
    chapters = re.split(r'(CHAPTER\s+[\w\d\.]+.*)', text)
    
    docs = []
    current_chapter = "Introduction/Preamble"
    
    for part in chapters:
        # If this part looks like a header, save it as the current metadata
        if part.strip().startswith("CHAPTER"):
            current_chapter = part.strip().split('\n')[0] # Keep just the first line as the title
        else:
            # If it's the body text, create a Document with the metadata
            if part.strip(): 
                docs.append(Document(page_content=part.strip(), metadata={"source": current_chapter}))
    
    return docs

# --- MAIN EXECUTION ---

# 1. Setup Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "text", "Three_Day_Pass.txt")

# 2. Load and Group by Chapter
print("Loading text and identifying chapters...")
raw_chapter_docs = load_and_split_by_chapter(file_path)

# 3. Chunk the text (smaller chunks for better precision)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=100,
    separators=["\n\n", ". "]
)

final_chunks = text_splitter.split_documents(raw_chapter_docs)
print(f"Created {len(final_chunks)} chunks with metadata.")

# 4. Save to Vector Store
print("Saving to database...")
db = Chroma.from_documents(
    documents=final_chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

print("Ingestion Complete! Database ready for querying.")