#WHAT THIS DOES:
#Complete RAG chatbot - load documents, ask questions, get grounded answers.
#Combines everything we have done earlier - embeddings + ChromaDB + chunking + RAG.
#Also adds conversation memory so agent can remember the session.

import os
import shutil
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory 
from dotenv import load_dotenv

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    print(f"[Setup] {len(documents)} docs -> {len(chunks)} chunks indexed")

    if os.path.exists("./chroma_db_chat"):
        shutil.rmtree("./chroma_db_chat")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory="./chroma_db_chat",
        collection_name="chat_docs"
    )    

    return vectorstore

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful document assistant.
        Answer questions based ONLY on the provided context documents.
        If the answer is not in the context, say so clearly.
        You also remember the conversation history shown below."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", """Context from documents:
        {context}
        Question: {question}""")
    ])    

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "history": lambda _: chat_history.messages
        }
        | prompt
        | llm
        | StrOutputParser()
    )   
    return chain 


knowledge_base = [
    Document(
        page_content="""LangGraph is a library for building stateful multi-agent
        applications. It models workflows as graphs with nodes and edges.
        State flows through every node. It supports cyclic graphs enabling
        agents to loop and retry. Built on top of LangChain.""",
        metadata={"source": "langgraph_docs"}
    ),
    Document(
        page_content="""RAG (Retrieval Augmented Generation) improves LLM accuracy
        by connecting models to external knowledge bases. Documents are
        chunked, embedded, and stored in a vector database. At query time,
        relevant chunks are retrieved and sent as context to the LLM.
        This grounds answers in real data and reduces hallucination.""",
        metadata={"source": "rag_guide"}
    ),
    Document(
        page_content="""ChromaDB is an open-source vector database for storing
        embeddings. It enables fast similarity search using L2 distance.
        Lower distance score means more similar. ChromaDB can run locally
        and integrates directly with LangChain via langchain-chroma.""",
        metadata={"source": "chromadb_docs"}
    ),
    Document(
        page_content="""FastAPI is a modern Python web framework for building APIs.
        It uses Python type hints for automatic validation. FastAPI supports
        async operations and generates automatic documentation at /docs.
        It is widely used to serve AI models and agents as web services.""",
        metadata={"source": "fastapi_docs"}
    ),
    Document(
        page_content="""Prompt engineering techniques include chain-of-thought,
        few-shot prompting, and system prompt design. Chain-of-thought
        forces step-by-step reasoning. Few-shot gives the model examples.
        System prompts control persona and output format.""",
        metadata={"source": "prompt_guide"}
    )
]     

def main():
    print("="*70)
    print("Document Q&A - RAG Chatbot")
    print("="*70)
    print("Loading knowledge base...")

    time.sleep(2)
    vectorstore = build_vectorstore(knowledge_base)
    time.sleep(2)
    chain = build_rag_chain(vectorstore)

    print("\nReady! Ask me anything about the documents.")
    print("Commands: 'quit' to exit, 'clear' to reset history\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                chat_history.clear()
                print("[History cleared]\n")
                continue

            time.sleep(1)
            answer = chain.invoke(user_input)
            print(f"\nAssistant: {answer}\n")

            chat_history.add_user_message(user_input)
            chat_history.add_ai_message(answer)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        except Exception as e:
            print(f"[Error: {e}]\n")


chat_history = ChatMessageHistory()

if __name__ == "__main__":
    main()



