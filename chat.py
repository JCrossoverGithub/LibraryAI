from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def start_chat_pipeline(chroma_dir: str):
    print("Connecting to local database...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cuda'}
    )
    
    vector_db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_model)
    memory_db = Chroma(collection_name="chat_memory", persist_directory=chroma_dir, embedding_function=embedding_model)

    print("Initializing qwen3-coder:30b model via Ollama...")
    llm = OllamaLLM(model="qwen3-coder:30b")

    # [NEW: 1] The Reformulation Prompt
    # This instructs the AI to ONLY rewrite the question, not answer it.
    rephrase_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. 
    Look at the chat history to understand the exact context (e.g., if the user says "it" or "simple search", figure out what they mean).
    ONLY output the standalone question. Do not answer it. Do not add conversational filler.

    Recent Chat History:
    {recent_history}

    Follow Up Input: {question}
    Standalone Question:"""
    
    REPHRASE_PROMPT = PromptTemplate.from_template(rephrase_template)
    rephrase_chain = REPHRASE_PROMPT | llm

    # Your existing QA Prompt
    qa_template = """You are a highly intelligent AI Library assistant. 
    If you don't know the answer, simply say "I cannot find the answer." Do not invent or hallucinate information.

    Past Conversation Context:
    {chat_history}

    Library Context: 
    {context}

    Question: {question}

    Helpful Answer:"""
    
    QA_PROMPT = PromptTemplate.from_template(qa_template)
    qa_chain = QA_PROMPT | llm

    print("\n✅ AI is ready to chat!")
    print("="*70)

    # [NEW: 2] Short-Term Memory Buffer
    # This keeps the chronological flow of the last 3 turns in RAM for the rephraser
    recent_chat_buffer = []

    # The Chat Loop
    while True:
        user_input = input("\nAsk your library a question (or type 'exit'): ")
        
        if user_input.lower() == 'exit':
            print("Shutting down...")
            break

        # ---------------------------------------------------------
        # [NEW: The Command Router]
        # ---------------------------------------------------------
        if user_input.strip().lower().startswith("-info "):
            fact_to_save = user_input[6:].strip()
            
            # 1. Save to Long-Term Chroma Memory
            memory_db.add_texts(
                texts=[f"User provided an explicit fact to remember: {fact_to_save}"],
                metadatas=[{"role": "user", "type": "explicit_fact"}]
            )
            
            print(f"[Memory Saved]: I will remember that: '{fact_to_save}'")
            
            # 2. [THE FIX] Inject it into the Short-Term Buffer!
            # This ensures the Rephraser immediately knows about the new fact.
            recent_chat_buffer.append(f"User explicitly stated a fact: {fact_to_save}")
            if len(recent_chat_buffer) > 6:
                recent_chat_buffer = recent_chat_buffer[-6:]
            
            continue
        # ---------------------------------------------------------

        print("Thinking...\n")
        
        # Format the short-term buffer into a single string
        recent_history_str = "\n".join(recent_chat_buffer) if recent_chat_buffer else "No recent conversation."

        # [NEW: 3] Execute the Reformulation Step
        # Pass the recent history and the raw input to get a mathematically precise search query
        standalone_query = rephrase_chain.invoke({
            "recent_history": recent_history_str,
            "question": user_input
        }).strip()
        
        print(f"🔍 [Debug - Searching DB for]: {standalone_query}\n")

        # Now, use the STANDALONE QUERY to search both databases
        memory_docs = memory_db.similarity_search(standalone_query, k=2)
        chat_history_str = "\n\n".join([doc.page_content for doc in memory_docs]) if memory_docs else "No relevant past conversations found."

        library_docs = vector_db.similarity_search(standalone_query, k=3)
        context_str = "\n\n".join([doc.page_content for doc in library_docs]) if library_docs else ""

        # Execute the final QA chain with all variables
        ai_answer = qa_chain.invoke({
            "question": standalone_query, # We use the standalone query here too for clarity
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

        # [NEW: 4] Update the Databases and Buffers
        # 1. Save to Long-Term Chroma Memory
        memory_db.add_texts(
            texts=[f"User asked: {user_input}", f"AI answered: {ai_answer}"],
            metadatas=[{"role": "user"}, {"role": "assistant"}]
        )
        
        # 2. Update Short-Term RAM Buffer (keep only the last 3 conversational turns to save tokens)
        recent_chat_buffer.append(f"User: {user_input}")
        recent_chat_buffer.append(f"AI: {ai_answer}")
        if len(recent_chat_buffer) > 6: # 3 turns = 6 messages (3 user, 3 AI)
            recent_chat_buffer = recent_chat_buffer[-6:]

if __name__ == "__main__":
    CHROMA_DIRECTORY = "./chroma_db"
    start_chat_pipeline(CHROMA_DIRECTORY)