from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def start_chat_pipeline(chroma_dir: str):
    """Connects the local DB to a local LLM to answer questions."""
    
    print("Connecting to local database...")
    # Updated to the new HuggingFaceEmbeddings class
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cuda'}
    )
    
    # 1. Your existing document library (Updated to new Chroma class)
    vector_db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embedding_model
    )

    # 2. Initialize the dedicated Memory Database
    memory_db = Chroma(
        collection_name="chat_memory",
        persist_directory=chroma_dir,
        embedding_function=embedding_model
    )

    print("Initializing qwen3-coder:30b model via Ollama...")
    # Updated to the new OllamaLLM class
    llm = OllamaLLM(model="qwen3-coder:30b")

    # 3. Create the System Prompt
    prompt_template = """You are a highly intelligent AI library assistant. 
    If you don't know the answer, simply say "I cannot find the answer." Do not invent or hallucinate information.

    Past Conversation Context:
    {chat_history}

    Library Context: 
    {context}

    Question: {question}

    Helpful Answer:"""
    
    PROMPT = PromptTemplate.from_template(prompt_template)

    # Combine the prompt and the LLM into a simple, modern chain
    chain = PROMPT | llm

    print("\n✅ AI is ready to chat!")
    print("="*70)

    # 4. The Chat Loop
    while True:
        user_input = input("\nAsk your library a question (or type 'exit'): ")
        
        if user_input.lower() == 'exit':
            print("Shutting down...")
            break
            
        print("Thinking...\n")
        
        # A. Fetch memory context
        memory_docs = memory_db.similarity_search(user_input, k=2)
        if memory_docs:
            chat_history_str = "\n\n".join([doc.page_content for doc in memory_docs])
        else:
            chat_history_str = "No relevant past conversations found."

        # B. Fetch library context manually (Replaces the rigid RetrievalQA chain)
        library_docs = vector_db.similarity_search(user_input, k=3)
        context_str = "\n\n".join([doc.page_content for doc in library_docs])

        # C. Execute the chain with all variables
        ai_answer = chain.invoke({
            "question": user_input,
            "context": context_str,
            "chat_history": chat_history_str
        })
        
        print("🤖 ANSWER:")
        print(ai_answer)
        
        print("\n📚 SOURCES USED:")
        if library_docs:
            for doc in library_docs:
                source_file = doc.metadata.get('source', 'Unknown')
                page_num = doc.metadata.get('page', 'Unknown')
                print(f"- {source_file} (Page {page_num})")
        else:
            print("- No documents retrieved.")
        print("-" * 70)

        # D. Save the new interaction to the memory database
        memory_db.add_texts(
            texts=[f"User asked: {user_input}", f"AI answered: {ai_answer}"],
            metadatas=[{"role": "user"}, {"role": "assistant"}]
        )

if __name__ == "__main__":
    CHROMA_DIRECTORY = "./chroma_db"
    start_chat_pipeline(CHROMA_DIRECTORY)