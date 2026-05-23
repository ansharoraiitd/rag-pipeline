#WHAT THIS DOES:
#ChromaDB stores embeddings and searches them by similarity.
#We add documents once, then query anytime - no re-embedding needed.
#This is the storage + retrieval layer of every RAG pipeline.

import time
import chromadb 
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document 
from dotenv import load_dotenv

load_dotenv()

#The embedding model:

embeddings_model = GoogleGenerativeAIEmbeddings (
    model = "models/gemini-embedding-001"
)

#Section1: create a ChromDB collection:
# A collection is like a table in a regular database.
# We store related documents in one collection.
print("="*50)
print("SECTION 1: Create a collection and add documents")
print("="*50)

#These are the documents we will store
#In a real RAG system, these would be chunks from the company's docs/PDFs
documents=[
    Document(
        page_content = "LangGraph is a library for building stateful multi-agent applications. It models agents as graphs where nodes are steps and edges connect them.",
        metadata = {"source": "langchain_docs", "topic": "langgraph"}
    ),
    Document(
        page_content = "RAG stands for Retrieval Augmented Generation. It connects LLMs to external knowledge bases to answer questions accurately.",
        metadata = {"source": "ai_glossary", "topic": "rag"}
    ),
    Document(
        page_content = "ChromaDB is a vector database designed for storing and searching embeddings. It enables semantic search across large document collections.",
        metadata = {"source": "chromadb_docs", "topic": "chromadb"}
    ),
    Document(
        page_content = "FastAPI is a modern web framework for building APIs with Python. It supports async operations and automatic documentation.",
        metadata = {"source": "fastapi_docs", "topic": "fastapi"}
    ),
    Document(
        page_content = "Embeddings are numerical representations of text that capture semantic meaning. Similar texts have similar embeddings.",
        metadata = {"source": "ml_concepts", "topic": "embeddings"}
    )
]

print(f"Adding {len(documents)} documents to ChromaDB...")

#Create a Chroma vector store
#persist_directory saves to disk so we don't lose data on restart.
#chroma_db/ is in .gitignore so it won't be committed.
time.sleep(1)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory="./chroma_db",
    collection_name="knowledge_base"
)

print("Documents stored successfully...")
print("ChromaDB saved to: ./chroma_db/")

#Section 2: search by similarity:
#similarity_search() takes a query string, embeds it, and returns the k most similar documents.
print("\n" + "=" * 50)
print("SECTION 2: Similarity search")
print("=" * 50)

time.sleep(2)

queries = [
    "How do I build AI agents?",
    "What is semantic search?",
    "How do I build a web API?"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    results = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(results):
        print(f"Result {i+1}: {doc.page_content[:80]}...")
        print(f"Source: {doc.metadata['topic']}")
    time.sleep(1) 

#Section 3: search with scores
#similarity_search_with_score() also returns how similar each result is
#lower score means more similar
print("\n" + "=" * 50)
print("SECTION 3: Search with similarity scores")
print("=" * 50)

time.sleep(2)

query = "What is RAG and how does it work?"
print(f"Query: '{query}'\n")

results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | Topic: {doc.metadata['topic']}")
    print(f"Content: {doc.page_content[:80]}...")
    print()

print("Lower score = more similar to query")

# ── Section 4: load existing collection from disk ────────────
print("=" * 50)
print("SECTION 4: Load existing collection from disk")
print("=" * 50)

# Load the same collection we created above
existing_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings_model,
    collection_name="knowledge_base"
)

# Do a quick search to prove data is still there
time.sleep(1)
test_results = existing_store.similarity_search("what is RAG?", k=1)
print(f"Test search returned: '{test_results[0].page_content[:60]}...'")
print(f"Data persists between runs — no re-embedding needed")

#Whenever trying to run the code: first type "rm -rf chroma_db/" and then "python 02_chromadb.py" in terminal
#Reason: Otherwise we will get duplicate results.
#On the duplicate results — this happens because we ran the file multiple times and ChromaDB added the same documents each time.
#The database now has duplicates.

