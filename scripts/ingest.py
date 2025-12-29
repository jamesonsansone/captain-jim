import os
import shutil
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# --- Configuration ---
DOCS_DIR = "./source_documents"  # Put 'Three_Day_Pass.txt' here
STORAGE_DIR = "./storage"        # Where the index will be saved
CHUNK_SIZE = 512                 # Smaller chunks for better accuracy
CHUNK_OVERLAP = 50               # Overlap to maintain context
MIN_CHAR_LENGTH = 50             # Filter out noise/short headers

def main():
    print(f"🚀 Starting ingestion from {DOCS_DIR}...")

    # 1. Setup FastEmbed (The Speed Upgrade)
    # This runs on CPU automatically and is ~50% faster than standard PyTorch
    print("🔹 Initializing FastEmbed model...")
    Settings.embed_model = FastEmbedEmbedding(
        model_name="BAAI/bge-small-en-v1.5", 
        max_length=512
    )
    
    # 2. Setup the Splitter (The Logic Upgrade)
    # We use a specific splitter to control chunk sizes strictly
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # We don't need an LLM for ingestion, so we set it to None to save resources
    Settings.llm = None

    # 3. Load Documents
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"❌ Created missing {DOCS_DIR}. Please put your .txt files there and run again.")
        return

    print("🔹 Loading documents...")
    documents = SimpleDirectoryReader(DOCS_DIR).load_data()
    print(f"   Loaded {len(documents)} source files.")

    # 4. Split and Clean (Removing Low Character responses)
    print("🔹 Splitting and cleaning nodes...")
    # Get nodes using the splitter defined in Settings
    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    
    initial_count = len(nodes)
    # Filter: Remove nodes that are too short (noise, page numbers, weird formatting)
    nodes = [node for node in nodes if len(node.get_content()) >= MIN_CHAR_LENGTH]
    removed_count = initial_count - len(n