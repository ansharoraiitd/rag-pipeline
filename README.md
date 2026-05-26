# RAG Pipeline

A complete Retrieval Augmented Generation system —
load documents, ask questions, get accurate grounded answers
with conversation memory.

## How it works
1. Documents split into overlapping chunks (300 chars, 50 overlap)
2. Each chunk embedded using Gemini embedding model
3. Embeddings stored in ChromaDB for fast similarity search
4. User question → embedded → ChromaDB finds relevant chunks
5. Chunks + question + conversation history → LLM → grounded answer
6. Model answers ONLY from retrieved context — no hallucination

## Main project: Document Q&A Chatbot
A CLI chatbot that lets you load any documents and have a
conversation about them. Remembers conversation history.
Refuses to answer questions not found in the documents.

Run it:
python chat_with_docs.py

## Files
| File | What it does |
|------|-------------|
| chat_with_docs.py | Main RAG chatbot — full pipeline |
| 01_embeddings.py | Embeddings — semantic similarity demo |
| 02_chromadb.py | ChromaDB — store and search embeddings |
| 03_chunking.py | Document chunking with overlap |
| 04_rag_chain.py | Full RAG chain — retrieval + generation |

## Key concepts
- Semantic search — find documents by meaning not keywords
- Chunking with overlap — no context lost at boundaries
- Grounding — model only answers from retrieved documents
- Conversation memory inside RAG pipeline

## Tech stack
Python · LangChain · ChromaDB · Gemini API · Embeddings

## Setup
pip install -r requirements.txt
Add GEMINI_API_KEY to .env
python chat_with_docs.py