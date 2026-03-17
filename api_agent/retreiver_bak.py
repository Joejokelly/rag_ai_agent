import os
import chromadb

from sentence_transformers import SentenceTransformer

print('retriever : before base_dir')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

print('retriever get_collection :: rag_docs')
collection = client.get_collection("rag_docs")

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_docs(query):
    print("RETRIEVE: started")

    query_embedding = model.encode([query])
    print("RAG AGENT: documents retrieved")



    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=2
    )

    print("RAG AGENT: documents retrieved")

    return results["documents"][0]

