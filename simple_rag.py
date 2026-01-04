# file: simple_rag.py
from openai import OpenAI
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_documents")

# Add sample docs
documents = [
    "Python is a great language for machine learning.",
    "Deep learning uses neural networks.",
    "JavaScript is often used for front-end development."
]

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents
)

def chat_with_docs(question: str) -> str:
    # 1. Search similar documents
    results = collection.query(
        query_texts=[question],
        n_results=2
    )
    context = "\n".join(results["documents"][0])

    # 2. Ask LLM with context
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer using only the context."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print(chat_with_docs("Which language is good for AI?"))
