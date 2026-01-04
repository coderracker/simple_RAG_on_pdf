from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI   
from langchain_classic.chains import RetrievalQA
from sentence_transformers import SentenceTransformer

#loading environment
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

#loading document
pdf_path = "samplepy.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

#chunking
splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 50
)
chunks = splitter.split_documents(documents)


#embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    chunks, 
    embedding = embeddings,
    persist_directory="./chroma_db"
)

#llm stuff
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

print("RAG system ready")


@app.route("/ask", methods = ["POST"])
def ask():
    data = request.json
    question = data.get("question","")
    if not question: 
        return jsonify({"error":"No questions provided"}), 400
    answer = qa.run(question)
    return jsonify({"question":question, "answer":answer})


@app.route("/", methods = ["GET"])
def index():
    return """
    <html>
    <head>
        <title>RAG Chatbot</title>
        <style>
            body { font-family: Arial; max-width: 700px; margin: 40px auto; }
            #chat { border: 1px solid #ccc; padding: 15px; height: 400px; overflow-y: scroll; }
            .msg { margin: 10px 0; padding: 8px; border-radius: 5px; }
            .user { background: #e3f2fd; text-align: right; }
            .bot { background: #f5f5f5; text-align: left; }
            input { width: 80%; padding: 10px; }
            button { padding: 10px 15px; margin-left: 5px; }
        </style>
    </head>
    <body>
        <h2> RAG Chatbot</h2>
        <div id="chat">
            <div class="msg bot">Hi! Ask me anything about the PDF.</div>
        </div>
        <input id="question" placeholder="Ask a question..." />
        <button onclick="askQuestion()">Send</button>
        <script>
            async function askQuestion() {
                const q = document.getElementById("question").value;
                if (!q) return;
                const chat = document.getElementById("chat");
                chat.innerHTML +=`<div class="msg user">${q}</div>`;
                document.getElementById("question").value ="";
                const res = await fetch("/ask", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({question: q})
                });
                const data = await res.json();
                chat.innerHTML+=`<div class="msg bot">${data.answer}</div>`
                chat.scrollTop = chat.scrollHeight;
            }
            document.getElementById("question").addEventListener("keypress", e => {
                if (e.key === "Enter") askQuestion();
            });
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    app.run(debug = True, host = "0.0.0.0", port = 8081)