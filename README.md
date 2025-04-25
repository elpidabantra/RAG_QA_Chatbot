# RAG_QA_Chatbot

# PDF Q&A Chatbot with RAG, LLaMa 2 & Streamlit Spaces

A Retrieval-Augmented Generation (RAG) chatbot that accepts user-uploaded PDF files, performs semantic retrieval over their content, and answers natural language questions using Meta’s LLaMa 2 (7B) model. The system is built in Python, prototyped in Google Colab, and deployed as a public Streamlit Space on Hugging Face.

## Project Description

This repository implements a RAG pipeline to:
1. **Ingest and pre-process PDFs** using PyMuPDF to extract and chunk text into manageable units :contentReference[oaicite:0]{index=0}.  
2. **Generate embeddings** for each chunk via a SentenceTransformer model and store them in FAISS for efficient similarity search :contentReference[oaicite:1]{index=1}.  
3. **Retrieve** the top-k relevant chunks at query time by embedding user questions and performing a nearest-neighbor search in the vector store :contentReference[oaicite:2]{index=2}.  
4. **Generate answers** by combining retrieved chunks with the user query into a prompt and feeding it to the quantized LLaMa 2 (7B) model :contentReference[oaicite:3]{index=3}.  
5. **Serve a web UI** built with Streamlit—allowing users to upload PDFs and chat—hosted as a public Space on Hugging Face for seamless access :contentReference[oaicite:4]{index=4}.

## Features

- **Document Loader & Chunking**: Splits PDFs into chunks of configurable token length for precise retrieval :contentReference[oaicite:5]{index=5}.  
- **Embedded Vector Store**: Uses FAISS for fast, scalable semantic search over document chunks :contentReference[oaicite:6]{index=6}.  
- **Quantized LLaMa 2 Inference**: Applies 4-bit AWQ quantization via bitsandbytes to run LLaMa 2 (7B) on limited-memory GPUs :contentReference[oaicite:7]{index=7}.  
- **Streamlit UI**: Interactive file uploader and chat interface, rerunning reactively on user input :contentReference[oaicite:8]{index=8}.  
- **Hugging Face Spaces Deployment**: Public URL with SSL, automatic rebuilds on GitHub push, and environment variable / secret management :contentReference[oaicite:9]{index=9}.

## Architecture

```mermaid
flowchart TD
  A[Upload PDF via Streamlit] --> B[Text Extraction & Chunking]
  B --> C[Embedding Generation (SentenceTransformers)]
  C --> D[FAISS Vector Index]
  E[User Question] --> F[Query Embedding]
  F --> D
  D --> G[Retrieve Top-k Chunks]
  G --> H[Prompt Assembly]
  H --> I[Quantized LLaMa 2 Inference]
  I --> J[Display Answer in UI]
