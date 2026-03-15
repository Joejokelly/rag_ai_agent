from fastapi import FastAPI
import os
import chromadb
from sentence_transformers import SentenceTransformer
from llm_api_backend.llm_client import ask_llm
from api_agent.rag_agent import rag_pipeline

print("1. File loaded ")
app = FastAPI()


print("2 FastAPI created")
# -----------------------------
# Absolute path to vector_store
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

# Connect tro Chroma
client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

# Access existing collection
collection = client.get_collection("rag_docs")

print('3. access existing collections')

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print('4. load model')



# -----------------------------
# Ask endpoint
# -----------------------------
@app.get("/ask")
def ask(question: str):

    print("5. /ask endpoint called")

    answer = rag_pipeline(question)

    print("6. returning response")

    return {"answer": answer}



