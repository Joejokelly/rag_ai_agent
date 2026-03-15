# RAG AI Agent

This project implements a Retrieval Augmented Generation (RAG) system using:

- FastAPI
- ChromaDB
- Sentence Transformers
- Ollama LLM

## Architecture

User Question
→ FastAPI API
→ Retriever (ChromaDB)
→ Context + Prompt
→ LLM
→ Answer