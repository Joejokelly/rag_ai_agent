import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

print('ingest : in')

client = chromadb.PersistentClient(path="../vector_store")
collection = client.get_or_create_collection("rag_docs")

with open("../data/sample.txt", "r") as f:
    text = f.read()

print("LLM CLIENT: response received")

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
print("ingest : created embeddings")
embeddings = model.encode(chunks)


print("storing docs in chroma")
print("sort")
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(chunks))]
)

print("Stored:", collection.count())
