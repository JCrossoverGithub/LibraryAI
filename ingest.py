import os
import re
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def clean_pdf_text(text: str) -> str:
    """Removes common PDF artifacts and noisy formatting."""
    
    # 1. Strip out hidden null bytes that often crash tokenizers
    text = text.replace('\x00', '')
    
    # 2. Remove standalone numbers on their own line (usually page numbers)
    # This regex looks for a line that contains only whitespace and digits
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 3. Fix words hyphenated across line breaks (e.g., "infor-\nmation" -> "information")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 4. Condense weird spacing (turn 3+ spaces or tabs into a single space)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 5. Condense massive gaps (turn 3+ empty lines into a standard double-newline paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def build_ingestion_pipeline(pdf_folder: str, chroma_persist_dir: str):
    """Reads PDFs, chunks text, embeds them, and stores them in ChromaDB."""
    
# --- 1. EXTRACT ---
    print(f"Loading PDFs from '{pdf_folder}'...")
    loader = PyPDFDirectoryLoader(pdf_folder)
    documents = loader.load()
    
    if not documents:
        print("No PDFs found. Please add some books to the directory.")
        return
    
    print(f"Successfully loaded {len(documents)} pages.")

    # --- 1.5. SANITIZE ---
    print("Scrubbing PDF artifacts and formatting noise...")
    for doc in documents:
        # Overwrite the dirty raw text with our cleaned text
        doc.page_content = clean_pdf_text(doc.page_content)

    # --- 2. CHUNK ---
    print("Chunking text with context awareness...")
    
    # RecursiveCharacterTextSplitter uses a hierarchy of separators to keep thoughts together.
    # It tries to split by double newlines (paragraphs) first. 
    # If a paragraph is still too big, it tries single newlines, then periods, then spaces.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # Note: This is measured in characters now, not tokens!
        chunk_overlap=200,   # A 200-character overlap (roughly 1-2 sentences)
        separators=["\n\n", "\n", "(?<=\. )", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split text into {len(chunks)} context-aware chunks.")

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