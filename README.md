#  AI Knowledge Twin

An AI-powered system that learns from your personal notes and provides intelligent, context-aware answers using Retrieval-Augmented Generation (RAG).

##  Features
- Upload PDF notes
- Semantic search using FAISS
- AI-powered answers using Groq LLM
- Personalized knowledge system
- Weak area detection based on user queries
- Fast and lightweight (no paid API required for embeddings)

##  Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API (LLM)

##  Demo Flow
1. Upload PDF notes
2. Text is extracted and split into chunks
3. Chunks are converted into embeddings
4. Stored in FAISS vector database
5. User asks a question
6. Relevant context is retrieved
7. Groq LLM generates a smart answer

## System Architecture
User → Streamlit UI → PDF Processing → Text Chunking
     → Embeddings (HuggingFace) → FAISS Vector DB
     → Context Retrieval → Groq LLM → Answer

## Use Cases
- Students for revision
- Personalized learning assistant
- Exam preparation
- Knowledge retrieval from notes

## Future improvements
- Chat history
- Multi-document support
- Advanced weak area analysis
- Offline LLM integration

---
