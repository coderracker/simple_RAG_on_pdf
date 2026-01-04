# RAG PDF Chatbot

A modern Retrieval-Augmented Generation (RAG) chatbot that answers questions about PDF documents using embeddings, vector search, and large language models.

## Features

- **PDF Ingestion**: Upload and process PDF documents
- **Semantic Search**: Uses OpenAI embeddings + Chroma vector store for intelligent document retrieval
- **AI-Powered Answers**: Leverages GPT-4o-mini to generate accurate responses grounded in your documents
- **Web Interface**: Simple, intuitive Flask-based UI for asking questions
- **Production-Ready**: Deployed on Vercel with proper error handling and logging

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Flask, Python 3.9+ |
| **Vector Store** | Chroma (in-memory) |
| **Embeddings** | Sentence Transformer |
| **LLM** | OpenAI GPT-4o-mini |
| **Frontend** | HTML5, Vanilla JavaScript |
| **Deployment** | Vercel Serverless |

## Prerequisites

- Python 3.9 or higher
- OpenAI API key (get one at https://platform.openai.com/api-keys)
- Git & GitHub account
- Vercel account (free at https://vercel.com)
  ## Setup

1. Clone the repository
2. Create a `.env` file in the project root
3. Add your API key:

OPENAI_API_KEY=your_api_key_here

4. Install dependencies
5. Run the app

