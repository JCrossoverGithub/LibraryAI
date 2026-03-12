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

    # 1. The Reformulation Prompt
    rephrase_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a highly specific standalone search query. 
    
    CRITICAL INSTRUCTIONS:
    - If the user uses pronouns like "I", "me", or "my", rewrite them as "the user" or "the user's".
    - Resolve any vague words like "it" or "that" based on the Recent Chat History.
    - ONLY output the standalone question. Do not answer it. Do not add conversational filler.

    Recent Chat History:
    {recent_history}

    Follow Up Input: {question}
    Standalone Question:"""
    
    REPHRASE_PROMPT = PromptTemplate.from_template(rephrase_template)
    rephrase_chain = REPHRASE_PROMPT | llm

    # 2. The QA Prompt
    qa_template = """You are a highly intelligent AI assistant. 
    You have access to two equal sources of truth:
    1. Past Conversation Context (memories and facts the user explicitly told you).
    2. Library Context (documents retrieved from the database).

    Answer the user's question using information from EITHER of these sources. If the user asks about a personal fact they previously shared, retrieve it from the Past Conversation Context.
    If you cannot find the answer in either source, simply say "I cannot find the answer." Do not invent or hallucinate information.

    Past Conversation Context:
    {chat_history}

    Library Context: 
    {context}

    Question: {question}

    Helpful Answer:"""
    
    QA_PROMPT = PromptTemplate.from_template(qa_template)
    qa_chain = QA_PROMPT | llm

    print("\n✅ AI is ready to chat!")
    print("Type '-info [fact]' to explicitly save a memory.")
    print("Type '-facts' to view saved facts, or '-forget [keyword]' to delete one.")
    print("Type '-strict [query]' for library-only, or '-chat [query]' for memory-only.")
    print("Type '-clear' to wipe short-term RAM, or '-wipe' for total memory wipe.")
    print("="*70)

    recent_chat_buffer = []

    while True:
        raw_user_input = input("\nAsk your library a question (or type 'exit'): ")
        user_input = raw_user_input.strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down...")
            break
            
        # ==========================================
        # COMMAND ROUTER (Direct Database Actions)
        # ==========================================
        
        # 1. -info : Save an explicit fact
        if user_input.lower().startswith("-info "):
            fact_to_save = user_input[6:].strip()
            memory_db.add_texts(
                texts=[f"User provided an explicit fact to remember: {fact_to_save}"],
                metadatas=[{"role": "user", "type": "explicit_fact"}]
            )
            print(f"🧠 [Memory Saved]: I will remember that: '{fact_to_save}'")
            recent_chat_buffer.append(f"User explicitly stated a fact: {fact_to_save}")
            if len(recent_chat_buffer) > 6:
                recent_chat_buffer = recent_chat_buffer[-6:]
            continue 

        # 2. -facts : Read all explicit facts
        elif user_input.lower() == "-facts":
            # Query chroma using the metadata filter
            all_facts = memory_db.get(where={"type": "explicit_fact"})
            print("\n🧠 [Explicit Facts Remembered]:")
            if not all_facts['documents']:
                print("  - No facts saved yet.")
            else:
                for doc in all_facts['documents']:
                    # Strip the prefix for cleaner display
                    clean_doc = doc.replace("User provided an explicit fact to remember: ", "")
                    print(f"  - {clean_doc}")
            continue

        # 3. -forget : Delete an explicit fact containing a keyword
        elif user_input.lower().startswith("-forget "):
            keyword = user_input[8:].strip().lower()
            all_facts = memory_db.get(where={"type": "explicit_fact"})
            ids_to_delete = []
            
            # Find which fact contains the keyword
            for doc_id, doc_text in zip(all_facts['ids'], all_facts['documents']):
                if keyword in doc_text.lower():
                    ids_to_delete.append(doc_id)
                    
            if ids_to_delete:
                memory_db._collection.delete(ids=ids_to_delete)
                print(f"🗑️ [Memory Deleted]: Forgot {len(ids_to_delete)} fact(s) containing '{keyword}'.")
            else:
                print(f"⚠️ [Not Found]: No explicit facts found containing '{keyword}'.")
            continue

        # 4. -clear : Wipe the short-term buffer
        elif user_input.lower() == "-clear":
            recent_chat_buffer.clear()
            print("🧹 [Buffer Cleared]: Short-term RAM wiped. The AI forgot the immediate conversation.")
            continue

        # 5. -wipe : The Nuclear Option
        elif user_input.lower() == "-wipe":
            all_memories = memory_db.get()
            if all_memories['ids']:
                memory_db._collection.delete(ids=all_memories['ids'])
            recent_chat_buffer.clear()
            print("☢️ [Nuclear Wipe]: All long-term and short-term chat memories have been permanently deleted.")
            continue

        # ==========================================
        # RAG ROUTING (Search Modifiers)
        # ==========================================
        
        use_library = True
        use_memory = True
        query_text = user_input

        # Force Library Only
        if user_input.lower().startswith("-strict "):
            use_memory = False
            query_text = user_input[8:].strip()
            print("📚 [Strict Mode]: Ignoring past conversations. Searching ONLY the library.")
            
        # Force Memory Only
        elif user_input.lower().startswith("-chat "):
            use_library = False
            query_text = user_input[6:].strip()
            print("💬 [Chat Mode]: Ignoring the library. Searching ONLY past conversations.")

        # ==========================================
        # CORE PIPELINE
        # ==========================================
        
        print("Thinking...\n")
        
        recent_history_str = "\n".join(recent_chat_buffer) if recent_chat_buffer else "No recent conversation."

        # The Reformulation Step uses `query_text` (without the command flags)
        standalone_query = rephrase_chain.invoke({
            "recent_history": recent_history_str,
            "question": query_text
        }).strip()
        
        print(f"🔍 [Debug - Searching DB for]: {standalone_query}\n")

        # Memory Search Execution
        if use_memory:
            memory_docs = memory_db.similarity_search(standalone_query, k=2)
            chat_history_str = "\n\n".join([doc.page_content for doc in memory_docs]) if memory_docs else "No relevant past conversations found."
        else:
            chat_history_str = "Memory search disabled by user (-strict mode)."

        # Library Search Execution
        if use_library:
            library_docs = vector_db.similarity_search(standalone_query, k=3)
            context_str = "\n\n".join([doc.page_content for doc in library_docs]) if library_docs else ""
        else:
            library_docs = []
            context_str = "Library search disabled by user (-chat mode)."

        # LLM Generation
        ai_answer = qa_chain.invoke({
            "question": standalone_query,
            "context": context_str,
            "chat_history": chat_history_str
        })
        
        print("🤖 ANSWER:")
        print(ai_answer)
        
        print("\n📚 SOURCES USED:")
        if library_docs and use_library:
            for doc in library_docs:
                source_file = doc.metadata.get('source', 'Unknown')
                page_num = doc.metadata.get('page', 'Unknown')
                print(f"- {source_file} (Page {page_num})")
        elif use_library:
            print("- No documents retrieved.")
        else:
            print("- Library ignored (-chat mode).")
        print("-" * 70)

        # Update Databases (only save standard queries, not strict/chat specific ones)
        if use_library and use_memory:
            memory_db.add_texts(
                texts=[f"User asked: {query_text}", f"AI answered: {ai_answer}"],
                metadatas=[{"role": "user"}, {"role": "assistant"}]
            )
        
        recent_chat_buffer.append(f"User: {query_text}")
        recent_chat_buffer.append(f"AI: {ai_answer}")
        if len(recent_chat_buffer) > 6:
            recent_chat_buffer = recent_chat_buffer[-6:]

if __name__ == "__main__":
    CHROMA_DIRECTORY = "./chroma_db"
    start_chat_pipeline(CHROMA_DIRECTORY)