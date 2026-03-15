from sentence_transformers import SentenceTransformer
import chromadb
from llm_api_backend.llm_client import ask_llm

print("RAG AGENT: module loaded")

client = chromadb.PersistentClient(path="vector_store")
collection = client.get_collection("rag_docs")

model = SentenceTransformer("all-MiniLM-L6-v2")


def rag_pipeline(question):

    print("RAG AGENT: pipeline started")

    query_embedding = model.encode([question])

    print("RAG AGENT: querying vector DB")

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=2
    )

    print("RAG AGENT: documents retrieved ->", results["documents"])

    context = " ".join(results["documents"][0])

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}
    """

    print("RAG AGENT: prompt built")

    answer = ask_llm(prompt)

    print("RAG AGENT: LLM response received")

    return answer


