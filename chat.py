from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

def start_chat_pipeline(chroma_dir: str):
    """Connects the local DB to a local LLM to answer questions."""
    
    # 1. Reconnect to your database using the exact same embedding model
    print("Connecting to local database...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embedding_model
    )

    # 2. Wake up your local AI
    print("Initializing local Mistral model via Ollama...")
    # This automatically connects to the Ollama server running on your machine
    llm = Ollama(model="mistral")

    # 3. Create the System Prompt
    # This dictates exactly how the AI should behave and format its answers.
    prompt_template = """You are a highly intelligent library assistant. Use the provided context extracted from books to answer the user's question accurately. 
    If you don't know the answer based strictly on the context, simply say "I cannot find the answer in the provided documents." Do not invent or hallucinate information.

    Context: {context}

    Question: {question}

    Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

# 4. Build the RAG Chain
    print("Building RAG chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # Update to MMR search type and add fetch_k
        retriever=vector_db.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 3, "fetch_k": 10}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    print("\n✅ AI is ready to chat! (Remember, local processing may take a moment to generate answers depending on your hardware.)")
    print("="*70)

    # 5. The Chat Loop
    while True:
        user_input = input("\nAsk your library a question (or type 'exit'): ")
        
        if user_input.lower() == 'exit':
            print("Shutting down...")
            break
            
        print("Thinking...\n")
        
        # This single line executes the entire pipeline: Search DB -> Inject Context -> Generate Answer
        response = qa_chain.invoke({"query": user_input})
        
        print("🤖 ANSWER:")
        print(response['result'])
        
        print("\n📚 SOURCES USED:")
        # Loop through the exact chunks it read to formulate that answer
        for doc in response['source_documents']:
            source_file = doc.metadata.get('source', 'Unknown')
            page_num = doc.metadata.get('page', 'Unknown')
            print(f"- {source_file} (Page {page_num})")
        print("-" * 70)

if __name__ == "__main__":
    CHROMA_DIRECTORY = "./chroma_db"
    start_chat_pipeline(CHROMA_DIRECTORY)