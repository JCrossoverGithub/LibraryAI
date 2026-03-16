# Personal Knowledge Assistant (LibraryAI)

A local AI assistant for querying your documents, remembering important user-provided facts, and falling back to live web search when local context is not enough.

## Overview

LibraryAI started as a document-based RAG project for a personal library. It has since evolved into a more complete local assistant that combines:

- semantic search over personal documents
- persistent conversational memory
- explicit fact storage
- follow-up question reformulation
- live web search fallback
- direct document management from the CLI

The result is a stronger prototype for a **local personalized AI assistant**.

## What the project does

The assistant can answer questions using three different context sources:

1. **Library context** — chunks retrieved from uploaded documents
2. **Conversation memory** — saved facts and prior interactions
3. **Live web search** — used directly with `-web` or automatically as a fallback

This makes the system a hybrid of:

- document retrieval
- long-term memory
- local LLM generation
- web-augmented answering

## Features

- **Interactive CLI assistant** for asking questions naturally
- **Local document ingestion** for `.pdf`, `.txt`, `.docx`, and `.doc`
- **Semantic retrieval** using a local Chroma vector store
- **Persistent memory** for:
  - explicit user facts
  - previous user questions
  - previous assistant responses
- **Follow-up query rewriting** to improve retrieval quality
- **Retrieval routing**:
  - combined memory + library search
  - library-only search
  - memory-only search
  - web-only search
- **Automatic web fallback** when local retrieval cannot answer
- **Source transparency** after answers
- **Library management** from the chat interface

## Architecture

### Core stack

- **Python**
- **Chroma** for vector storage
- **Hugging Face embeddings**
- **Ollama** for local LLM inference
- **LangChain prompt pipelines**
- **DuckDuckGo search**
- **PyPDF / text / DOCX loaders**
- **Recursive text chunking**

### Current runtime assumptions

The current implementation uses:

- `BAAI/bge-large-en-v1.5` for embeddings
- `qwen3-coder:30b` through Ollama (Interchangeable)
- a local Chroma persistence directory at `./chroma_db`
- CUDA for the current embedding model configuration

## How it works

When a user asks a question, the assistant:

1. rewrites the question into a standalone search query
2. checks recent conversation context
3. searches long-term chat memory
4. searches the document library or live web results
5. injects retrieved context into the QA prompt
6. generates an answer with a local Ollama-hosted model
7. stores the interaction back into memory when appropriate

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
| `-upload [path]` | Ingest a `.pdf`, `.txt`, `.docx`, or `.doc` file |
| `-docs` | List all documents in the library |
| `-remove [name]` | Delete a document from the library |

### Chat / retrieval commands

| Command | Purpose |
|---|---|
| `-strict [query]` | Search only the library |
| `-chat [query]` | Search only past conversations |
| `-web [query]` | Search only the live internet |
| `-clear` | Clear short-term recent chat memory |
| `-wipe` | Delete all stored chat memory |
| `exit` / `quit` | Close the application |

## Setup

> Note: this project currently assumes a local Ollama setup and a configured Python environment.
> A future version will include a `requirements.txt` or `pyproject.toml` for easier installation.

### Recommended prerequisites

- Python 3.10+
- Ollama installed and running locally
- the required Ollama model pulled locally
- a CUDA-capable GPU if you keep the current embedding configuration

### Run

```bash
python js_ai.py
```

## Example workflow

```text
-upload my_notes.pdf
-info I am focusing on distributed systems
What have I told you about my interests?
-strict What does the uploaded document say about vector databases?
-web latest news on local LLM tooling
```

## Why I built this

I wanted to build something more useful than a one question long "chat" utilizing knowledge from a PDF.
Something that I will actually use when researching for my future projects. And that is exactly what this turned out to be.

This project explores the combination of:

- personal knowledge retrieval
- persistent memory
- local-first AI tooling
- user-controlled document ingestion
- practical CLI workflows

It is meant to demonstrate applied experience with RAG, vector databases, local LLM pipelines, and AI assistant design.

## Project status

This is an actively evolving prototype. Current focus areas include:

- improving setup and reproducibility
- improving source formatting
- making the assistant easier to configure
- continuing the shift from virtual library tool to personal knowledge assistant

## Future improvements

- add a `requirements.txt` or `pyproject.toml`
- add configuration for model names and device selection
- add better source citation formatting
- add a GUI or web interface
- add ingestion metrics and document stats
- improve memory controls and retrieval ranking
