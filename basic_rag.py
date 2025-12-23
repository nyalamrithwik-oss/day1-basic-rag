# basic_rag.py
# Basic RAG System with ChromaDB

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = "./chroma_db"

class BasicRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        self.vectorstore = None
    
    def load_documents(self, file_paths):
        """Load documents from file paths"""
        documents = []
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                print(f"Skipping unsupported file: {file_path}")
                continue
            documents.extend(loader.load())
        return documents
    
    def chunk_documents(self, documents):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    
    def create_vectorstore(self, chunks):
        """Create vector store from chunks"""
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        print(f"✅ Created vector store with {len(chunks)} chunks")
    
    def query(self, question):
        """Query the RAG system"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized!")
        
        # Get relevant documents
        retriever = self.vectorstore.as_retriever()
        relevant_docs = retriever.invoke(question)
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt and get answer
        prompt = f"""Use the following context to answer the question.
        
Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.invoke(prompt)
        
        return {
            "answer": answer.content,
            "sources": relevant_docs
        }

# Test the system
if __name__ == "__main__":
    print("🚀 Starting Basic RAG System...")
    
    # Initialize RAG
    rag = BasicRAG()
    
    # Load documents
    print("\n📄 Loading documents...")
    docs = rag.load_documents([
        "test_document.txt",
    ])
    print(f"✅ Loaded {len(docs)} documents")
    
    # Chunk documents
    print("\n✂️ Chunking documents...")
    chunks = rag.chunk_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Create vector store
    print("\n🗄️ Creating vector store...")
    rag.create_vectorstore(chunks)
    
    # Query the system
    print("\n❓ Querying the system...")
    question = "What is this document about?"
    result = rag.query(question)
    
    print(f"\n📝 Question: {question}")
    print(f"\n💡 Answer: {result['answer']}")
    print(f"\n📚 Number of sources: {len(result['sources'])}")