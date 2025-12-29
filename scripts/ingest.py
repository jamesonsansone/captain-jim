import os
import shutil
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# --- Configuration ---
DOCS_DIR = "./source_documents"
STORAGE_DIR = "./storage" # Ensure this matches where server.py looks (e.g., chroma_db or storage)
# Note: If server.py looks at "chroma_db", you should point this there or update server.py

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHAR_LENGTH = 100  # <--- UPDATED: Increased to filter out headers/empty lines

def main():
    print(f"🚀 Starting ingestion from {DOCS_DIR}...")

    print("🔹 Initializing FastEmbed model...")
    Settings.embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-small-en-v1.5", 
        max_length=512
    )
    
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    Settings.llm = None

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"❌ Created missing {DOCS_DIR}. Please put your .txt files there.")
        return

    print("🔹 Loading documents...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    
    print("🔹 Splitting and cleaning nodes...")
    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    
    initial_count = len(nodes)
    # Filter: Remove nodes that are too short
    nodes = [node for node in nodes if len(node.get_content()) >= MIN_CHAR_LENGTH]
    removed_count = initial_count - len(nodes)
    
    print(f"   Created {len(nodes)} chunks (Removed {removed_count} small/noisy chunks).")

    print("🔹 Building Vector Index...")
    # show_progress=True to see the bar!
    index = VectorStoreIndex(nodes, show_progress=True)

    print(f"🔹 Persisting index to {STORAGE_DIR}...")
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)
        
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    main()