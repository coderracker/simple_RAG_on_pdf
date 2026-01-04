import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI   
from langchain_classic.chains import RetrievalQA


load_dotenv()

pdf_path = "samplepy.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"Loaded {len(documents)} pages")



splitter = CharacterTextSplitter(
    chunk_size =100,
    chunk_overlap = 50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---\n")
    print(chunk.page_content)
