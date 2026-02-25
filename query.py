from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def search_database(query: str, chroma_dir: str):
    """Searches the local ChromaDB for text chunks matching the query."""
    
    # 1. Initialize the exact same embedding model used for ingestion
    # If you use a different model here, the math won't match and the search will fail!
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    # 2. Connect to your existing local Chroma database
    vector_db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embedding_model
    )

    # 3. Perform the semantic search
    # k=3 tells the database to return the top 3 most relevant chunks
    results = vector_db.similarity_search(query, k=3)

    if not results:
        print("No relevant results found.")
        return

    # 4. Display the results and their metadata (Source file and Page number)
    print("\n" + "="*50)
    print(f"TOP 3 MATCHES FOR: '{query}'")
    print("="*50)
    
    for i, doc in enumerate(results, 1):
        # Extract metadata automatically saved during ingestion
        source_file = doc.metadata.get('source', 'Unknown File')
        page_num = doc.metadata.get('page', 'Unknown Page')
        
        print(f"\n--- Result {i} ---")
        print(f"Source: {source_file} (Page {page_num})")
        print(f"Excerpt: {doc.page_content[:300]}...\n") # Prints the first 300 characters of the chunk

if __name__ == "__main__":
    CHROMA_DIRECTORY = "./chroma_db"
    
    # Creates an interactive loop so you can ask multiple questions without restarting the script
    print("\n📚 LibraryAI Search Engine Started!")
    while True:
        user_input = input("\nWhat would you like to search for? (or type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            print("Shutting down search...")
            break
            
        search_database(user_input, CHROMA_DIRECTORY)