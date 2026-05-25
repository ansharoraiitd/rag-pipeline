#WHAT THIS DOES:
#Full RAG pipeline - index documents, retrieve relevant chunks, generate answers grounded in the retrieved content.
#This is the complete pattern used in every production RAG system.

import time
import shutil 
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

#Step 1: indexing documents
#This is the "indexing" phase - done once when documents are uploaded.
#In production, this runs when a user uploads a file, not on every query.

print("="*70)
print("STEP 1: Indexing documents")
print("="*70)

#Our knowledge base - later, this will come from real files.

documents = [
    Document(
        page_content="""LangGraph is a library for building stateful multi-agent
        applications with LLMs. It models workflows as graphs where nodes
        are computation steps and edges are transitions. State flows through
        every node. LangGraph supports cyclic graphs, which enables agents
        to loop and retry. It is built on top of LangChain.""",
        metadata={"source": "langgraph_docs"}
    ),
    Document(
        page_content="""RAG (Retrieval Augmented Generation) improves LLM accuracy
        by connecting models to external knowledge. The indexing phase splits
        documents into chunks and stores their embeddings in a vector database.
        The retrieval phase embeds the user query and finds similar chunks.
        The generation phase sends chunks as context to the LLM.""",
        metadata={"source": "rag_guide"}
    ),
      Document(
        page_content="""FastAPI is a modern Python web framework for building APIs.
        It is based on standard Python type hints. FastAPI supports async
        operations natively. It generates automatic interactive documentation.
        FastAPI is commonly used to serve ML models and AI agents as APIs.""",
        metadata={"source": "fastapi_docs"}
    ),
      Document(
        page_content="""ChromaDB is an open-source embedding database. It stores
        vector embeddings and enables fast similarity search. ChromaDB can
        run in-memory or persist to disk. It integrates with LangChain via
        the langchain-chroma package. ChromaDB uses L2 distance by default
        where lower score means more similar.""",
        metadata={"source": "chromadb_docs"}
    ),
      Document(
        page_content="""Prompt engineering is the practice of designing inputs to
        LLMs to get better outputs. Key techniques include chain-of-thought
        prompting, few-shot examples, and clear system instructions. The
        system prompt controls the model's persona and output format.
        Good prompt engineering can improve accuracy significantly.""",
        metadata={"source": "prompt_guide"}
    )
]

#Chunk the documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
print(f"Documents: {len(documents)} -> Chunks: {len(chunks)}")

#Store in ChromaDB — clear old data first for clean run
if os.path.exists("./chroma_db_rag"):
    shutil.rmtree("./chroma_db_rag")

time.sleep(2)

vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="./chroma_db_rag",
    collection_name="rag_knowledge"
)

print("Indexed and stored in ChromaDB")

#Step 2: building the RAG chain
#This is the core of RAG - connect retrieval to generation
#The retriever fetches relevant chunks from ChromaDB.
#The prompt template injects those chunks as context.
#The model generates an answer grounded in the context.

print("\n" + "="*70)
print("STEP 2: Building the RAG chain")
print("="*70)

# Retriever - wraps ChromaDB with a clean interface
# k=3 means return the most relevant 3 chunks
retriever = vectorstore.as_retriever(search_kwars={"k": 3})

#The RAG prompt - {context} gets filled with retrieved chunks.
# {question} gets filled with the user's query
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions
    based ONLY on the provided context. If the answer is not in the
    context, say "I don't have information about that in my knowledge base."
    Do not use any outside knowledge."""),
    ("human", """Context: {context}
    Question: {question}
    Answer based only on the context above:""")
])

def format_docs(docs):
    """Join retrieved chunks into one context string."""
    return "\n\n".join(doc.page_content for doc in docs)

# The RAG chain - LCEL pipe syntax from earlier.
# retriever gets the relevant docs
# RunnablePassThrough passes the question through unchanged
# format_docs joins the chunk into one string.

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | model
    | StrOutputParser()
)

print("RAG chain built successfully!")


# Step 3: ask questions
print("\n" + "=" * 50)
print("STEP 3: Asking questions")
print("=" * 50)

def ask(question: str):
    print(f"\nQ: {question}")
    time.sleep(1)
    answer = rag_chain.invoke(question)
    print(f"A: {answer}")
    print()

# Questions the knowledge base can answer
ask("What is LangGraph and what is it used for?")
ask("How does the RAG indexing phase work?")
ask("What distance metric does ChromaDB use?")

# Question the knowledge base cannot answer — tests grounding
ask("What is the capital of France?")