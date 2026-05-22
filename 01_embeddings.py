#WHAT THIS DOES:
#Embeddings turn text into lists of numbers representing meaning.
#Similar texts get similar numbers - this enables semantic search.
#This is the foundation of RAG -  you embed documents and queries, then find documents whose numbers are closest to the query numbers.

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np 

load_dotenv()

#Creating the embedding model:
#This calls the Gemini API to convert text -> numbers
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

#Section 1: embed a single piece of text
print("="*70)
print("SECTION 1: What does an embedding look like?")
print("="*70)

text = "LangGraph is a framework for building AI agents"
embedding = embeddings_model.embed_query(text)

print(f"Text: '{text}'")
print(f"Embedding type: {type(embedding)}")
print(f"Embedding length: {len(embedding)} numbers")
print(f"First 5 numbers: {embedding[:5]}")
print("These numbers represent the MEANING of the text.")


#Section 2: similar texts get similar embeddings
print("\n"+ "="*70)
print("SECTION 2: similar meaning = similar numbers")
print("="*70)

texts = [
    "LangGraph is used for building AI agents.",                #similar to query
    "Agents are built using the LangGraph framework.",          #similar to query
    "I enjoy eating pizza on Fridays.",                         #very different
    "RAG connects LLMs to external knowledge."                  #somewhat different
]

#Embed all texts at once:
text_embeddings = embeddings_model.embed_documents(texts)

#Embed a query:
query = "What is LangGraph?"
query_embedding = embeddings_model.embed_query(query)

#Calculating similarity - closer to 1.0 = more similar
#We use dot product as a simple similarity measure
print(f"Query: '{query}'\n")
for text, emb in zip(texts, text_embeddings):
    # numpy dot product between query and each text embedding
    similarity = np.dot(query_embedding, emb)
    print(f"Similarity: {similarity:.3f} | Text: '{text[:50]}")

print("\nHigher number = more similar meaning to the query")
print("This is exactly how RAG finds relevant documents")


#Section 3: the key insight
print("\n" + "="*70)
print("SECTION 3: Why this matters for RAG")
print("="*70)

print("""
RAG pipeline in plain English:

1. INDEXING (done once):
   - Take your documents
   - Embed every chunk → list of numbers
   - Store in ChromaDB (a database for embeddings)

2. RETRIEVAL (done every query):
   - User asks a question
   - Embed the question → list of numbers
   - Find the chunks with the most similar numbers
   - Return those chunks

3. GENERATION:
   - Send the relevant chunks + question to the LLM
   - LLM answers using the chunks as context

Next we will build the ChromaDB part (step 1 + 2).
After that, we will build the full pipeline (all 3 steps).
""")




