from llm_api_backend.llm_client import ask_llm
from api_agent.retriever import get_relevant_docs

print("RAG AGENT: module loaded")


def rag_pipeline(question: str):
    print("RAG AGENT: pipeline started")

    # --- Step 1: Retrieve docs ---
    docs = get_relevant_docs(question, k=2)

    print(f"RAG AGENT: retrieved {len(docs)} docs")

    # --- Step 2: Build context ---
    context = "\n".join(docs)

    # --- Step 3: Build prompt ---
    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

    print("RAG AGENT: prompt built")

    # --- Step 4: Call LLM ---
    answer = ask_llm(prompt)

    print("RAG AGENT: LLM response received")

    return answer
