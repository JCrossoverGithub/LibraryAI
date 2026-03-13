# LibraryAI

**LibraryAI** is a local, retrieval-augmented AI assistant for interacting with a personal document library. It allows a user to upload documents, semantically search them, ask natural-language questions, and combine retrieved document context with lightweight conversational memory.

The centerpiece of the project is **`chat.py`**, which turns the system into an interactive command-line assistant. Instead of being just a one-off PDF question-answering script, LibraryAI evolved into a more capable local assistant that supports:

- semantic search over a Chroma vector database
- persistent long-term chat memory
- explicit fact storage
- follow-up question reformulation
- document upload from the chat interface
- library management commands
- memory-only and library-only retrieval modes

This project is intended to demonstrate practical experience with:

- **retrieval-augmented generation (RAG)**
- **vector databases**
- **local LLM workflows**
- **embedding-based search**
- **CLI application design**
- **iterative feature-driven development**

---

## Why I built this

I wanted to build a local AI tool that feels more like a usable system than a basic demo. Many document-chat projects stop at “upload a PDF and ask a question.” This project goes further by adding:

- a persistent local knowledge store
- a conversational interface
- explicit memory controls
- direct ingestion from the chat loop
- multiple retrieval modes depending on the user’s intent

The result is a stronger prototype for a real personal knowledge assistant: something closer to a “virtual library brain” than a simple document lookup script.

---

## Core idea

LibraryAI uses a **local vector database** to store embedded chunks of documents and retrieve the most relevant passages for a user’s question.

In `chat.py`, the assistant also maintains a separate memory collection for:

- facts the user explicitly asks it to remember
- previous user queries
- previous assistant responses

When a user asks a question, the system:

1. rewrites follow-up questions into a more specific standalone query
2. searches past conversation memory
3. searches the document library
4. injects the retrieved context into a prompt
5. generates an answer with a local Ollama-hosted model

That makes the system a **hybrid of document retrieval + conversational memory**, rather than plain RAG alone.

---

## Main file: `chat.py`

`chat.py` is the primary application entry point and the most important file in the project.

### What it does

- connects to a local **Chroma** database
- loads a **HuggingFace embedding model**
- initializes an **Ollama LLM**
- supports document upload for `.pdf`, `.txt`, and `.docx`
- stores a second Chroma collection for chat memory
- reformulates ambiguous follow-up questions into standalone search queries
- retrieves relevant memory and library context
- answers questions using retrieved context only
- supports direct CLI commands for memory and library management

### Current architecture

The current `chat.py` uses:

- **Chroma** for vector storage
- **HuggingFace embeddings**
- **Ollama** for local LLM inference
- **LangChain prompt templates**
- **PyPDFLoader / TextLoader / Docx2txtLoader**
- **RecursiveCharacterTextSplitter**

---

## Features

### 1. Interactive local AI assistant
The project runs as a command-line chatbot for querying a personal document library.

### 2. Local document ingestion
Users can upload supported files directly from the chat interface instead of relying only on a separate ingestion script.

### 3. Semantic retrieval
Questions are answered using semantically similar chunks retrieved from the vector database.

### 4. Persistent memory
The assistant keeps a separate memory store for:
- explicit user facts
- prior user questions
- prior assistant answers

### 5. Question reformulation
Follow-up questions are rewritten into standalone search queries to improve retrieval quality.

### 6. Retrieval routing
The user can search:
- both memory and library
- only the library
- only past conversation memory

### 7. Source transparency
The system prints the document sources used for retrieval after answering.

### 8. Library management from the CLI
The chat interface also supports listing, removing, and uploading documents.

---

## Commands

### Memory commands

| Command | Purpose |
|---|---|
| `-info [fact]` | Save a personal fact |
| `-facts` | List saved facts |
| `-forget [word]` | Delete saved facts containing a keyword |

### Library commands

| Command | Purpose |
|---|---|
| `-upload [path]` | Ingest a `.pdf`, `.txt`, or `.docx` file |
| `-docs` | List all documents in the library |
| `-remove [name]` | Delete a file from the library |

### Chat commands

| Command | Purpose |
|---|---|
| `-strict [query]` | Search only the library |
| `-chat [query]` | Search only past conversations |
| `-clear` | Clear the short-term recent-history buffer |
| `-wipe` | Delete all long-term and short-term chat memory |
| `exit` / `quit` | Close the application |

---

## Example usage

```bash
python chat.py
