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

#loading pdf
pdf_path = "samplepy.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"Loaded {len(documents)} pages")


#chunking
splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

#embedding
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    chunks, 
    embedding = embeddings,
    persist_directory="./chroma_db"
)
print("vector store created")

#llm stuff
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

#query augmentation
print("\n" + "="*50)
print("RAG System Ready! Ask questions about your PDF.")
print("Type 'exit' to quit.")
print("="*50 + "\n")

while True:
    question = input("Your question: ").strip()
    if question.lower() in ["exit", "quit"]:
        break
    answer = qa.run(question)
    print(f"\nAnswer: {answer}\n")




