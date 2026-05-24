# WHAT THIS DOES:
# Documents are too long to embed whole - split into chunks first.
# chunk_size controls how big each piece is.
# chunk_overlap prevents losing meaning at the boundaries.
# This is the preprocessing step before storing in ChromaDB.

import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

#SECTION 1: BASIC CHUNKING
#RecursiveCharacterTextSplitter is the most commonly used splitter.
#It tries to split on paragraphs first, then sentences, then words.
#This keeps natural language boundaries intact where possible.

print("="*70)
print("SECTION 1: How chunking works")
print("="*70)

# A long piece of text to chunk:

long_text = """
LangChain is a framework for developing applications powered by language models.
It provides tools for chaining together different components like prompts, models,
and output parsers. LangChain makes it easy to build complex AI applications.

LangGraph is an extension of LangChain for building stateful multi-agent systems.
It models agent workflows as graphs where nodes represent steps and edges represent
transitions between steps. LangGraph enables building agents that can loop, branch,
and maintain state across multiple interactions.

RAG stands for Retrieval Augmented Generation. It is a technique that combines
information retrieval with text generation. RAG systems first retrieve relevant
documents from a knowledge base, then use those documents as context for the LLM
to generate accurate, grounded answers. This prevents hallucination.

ChromaDB is an open-source vector database. It stores embeddings and enables
fast similarity search. ChromaDB is commonly used as the retrieval layer in RAG
pipelines. It can run locally or in the cloud.
"""

# Create the splitter: 
# chunk_size: max characters per chunk
# chunk_overlap: how many characters to repeat between chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    length_function=len
)

chunks = splitter.split_text(long_text)

print(f"Original text length: {len(long_text)} characters")
print(f"Number of chunks: {len(chunks)}")
print(f"Chunk size setting: 200 chars | Overlap: 40 chars")
print()

for i, chunk in enumerate(chunks):
    print(f"Chunk{i+1}: {len(chunk)} chars")
    print(f"'{chunk[:50]}...'")
    print()


# SECTION 2: chunking with overlap
#Showing why overlap matters - demonstrating the boundary sharing
print("="*70)
print("SECTION 2: Why overlap matters")
print("="*70)

short_text="The quick brown fox jumps over the lazy dog. This is a test sentence for chunking. I want to see how the overlap works between chunks."

splitter_without_overlap = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=0
)

splitter_with_overlap = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=20
)

chunks_no_overlap = splitter_without_overlap.split_text(short_text)
chunks_with_overlap = splitter_with_overlap.split_text(short_text)

print("WITHOUT OVERLAP:")
for i,c in enumerate(chunks_no_overlap):
    print(f"Chunk {i+1}: '{c}'")

print("\nWITH overlap (20 chars):")
for i, c in enumerate(chunks_with_overlap):
    print(f"Chunk {i+1}: '{c}'")

print("\nNotice: with overlap, each chunk shares some text with the next.")
print("This means context is never completely cut off at a boundary.")

#SECTION 3: chunk documents and store in ChromaDB
#This is what happens in a real RAG pipeline.
# document → split into chunks → embed each chunk → store in ChromaDB
print("\n" + "=" * 50)
print("SECTION 3: Chunk real documents and store in ChromaDB")
print("=" * 50)

# Simulate real documents (in week 5 these will be actual PDFs)
raw_documents = [
    Document(
        page_content="""LangGraph is a library built on top of LangChain for building
stateful, multi-actor applications with LLMs. It extends LangChain's
capabilities by providing first-class support for cyclic computation,
which is essential for most agentic architectures. LangGraph models
workflows as graphs where nodes represent computation steps and edges
represent the flow of data between steps. The state flows through
every node, enabling each step to read what happened before.""",
        metadata={"source": "langgraph_guide", "page": 1}
    ),
    Document(
        page_content="""RAG (Retrieval Augmented Generation) is a technique that improves
LLM responses by connecting them to external knowledge. The process
has two phases: indexing and retrieval. During indexing, documents
are split into chunks, embedded into vectors, and stored in a vector
database. During retrieval, the user query is embedded and the most
similar document chunks are fetched. These chunks are then provided
as context to the LLM when generating its answer.""",
        metadata={"source": "rag_explainer", "page": 1}
    ),
]

# Split documents into chunks
doc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunked_docs = doc_splitter.split_documents(raw_documents)

print(f"Original documents: {len(raw_documents)}")
print(f"After chunking: {len(chunked_docs)} chunks")
print()

for i, chunk in enumerate(chunked_docs):
    print(f"Chunk {i+1}:")
    print(f"  Source: {chunk.metadata['source']}")
    print(f"  Length: {len(chunk.page_content)} chars")
    print(f"  Content: {chunk.page_content[:80]}...")
    print()

# Store chunks in ChromaDB
import shutil
import os
if os.path.exists("./chroma_db_chunks"):
    shutil.rmtree("./chroma_db_chunks")

print("Storing chunks in ChromaDB...")
time.sleep(1)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embeddings_model,
    persist_directory="./chroma_db_chunks",
    collection_name="chunked_docs"
)

time.sleep(2)

# Search the chunked store
query = "How does RAG work?"
results = vectorstore.similarity_search(query, k=2)
print(f"Query: '{query}'")
for i, doc in enumerate(results):
    print(f"Result {i+1}: {doc.page_content[:100]}...")
    print(f"Source: {doc.metadata['source']}")
