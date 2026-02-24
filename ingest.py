import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_ingestion_pipeline(pdf_folder: str, chroma_persist_dir: str):
    """Reads PDFs, chunks text, embeds them, and stores them in ChromaDB."""
    
    # --- 1. EXTRACT ---
    print(f"Loading PDFs from '{pdf_folder}'...")
    # PyPDFDirectoryLoader automatically processes all PDFs in the given folder
    loader = PyPDFDirectoryLoader(pdf_folder)
    documents = loader.load()
    
    if not documents:
        print("No PDFs found. Please add some books to the directory.")
        return
    
    print(f"Successfully loaded {len(documents)} pages.")

    # --- 2. CHUNK ---
    print("Chunking text...")
    # TokenTextSplitter ensures chunks are measured by tokens, not just characters.
    # 800 tokens with a 100 token overlap hits your 400-900 token / 10-20% overlap target.
    text_splitter = TokenTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        disallowed_special=()  # <--- Add this line to bypass the error
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split text into {len(chunks)} manageable chunks.")

    # --- 3. EMBED & STORE ---
    print("Initializing local embedding model...")
    # Using a fast, highly-rated open source embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Embedding and storing chunks into local ChromaDB at '{chroma_persist_dir}'...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=chroma_persist_dir
    )
    
    # Ensure the database is saved to disk
    vector_db.persist()
    print("✅ Ingestion pipeline complete! Your books are ready to be queried.")

if __name__ == "__main__":
    # Define your local paths
    PDF_DIRECTORY = "./data/books"
    CHROMA_DIRECTORY = "./chroma_db"
    
    # Create the PDF directory if it doesn't exist so you can drop files into it
    os.makedirs(PDF_DIRECTORY, exist_ok=True)
    
    build_ingestion_pipeline(PDF_DIRECTORY, CHROMA_DIRECTORY)