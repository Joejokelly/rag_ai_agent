from sentence_transformers import SentenceTransformer
import chromadb

print('rag_Agent in')

client = chromadb.PersistentClient(path="vector_store")
collection = client.get_collection("rag_docs")

model = SentenceTransformer("all-MiniLM-L6-v2")

print('rag_agent, what is rag')
question = "What is RAG?"

query_embedding = model.encode([question])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=2
)

context = " ".join(results["documents"][0])

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}
"""
print("RAG AGENT: documents retrieved")
print(prompt)