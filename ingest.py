import os
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import Chroma

# --- Configuration ---
KNOWLEDGE_DIR = "knowledge"
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "floatchat")

def main():
    """Reads knowledge files, splits them, and ingests them into ChromaDB."""
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        return

    print("--- Starting Knowledge Ingestion ---")
    
    # 1. Initialize Chroma client and embedding function
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = Chroma(
        client=chroma_client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding_function
    )
    
    # 2. Find and read knowledge files
    all_docs = []
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    for filename in os.listdir(KNOWLEDGE_DIR):
        if filename.endswith(".md"):
            filepath = os.path.join(KNOWLEDGE_DIR, filename)
            with open(filepath, 'r') as f:
                content = f.read()
            
            docs = text_splitter.create_documents([content])
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
            print(f"Loaded and split {len(docs)} documents from {filename}")

    # 3. Add documents to the vector store
    if all_docs:
        print(f"\nAdding {len(all_docs)} documents to Chroma collection '{CHROMA_COLLECTION}'...")
        vector_store.add_documents(documents=all_docs)
        print("--- Ingestion Complete ---")
    else:
        print("No documents found to ingest.")

if __name__ == "__main__":
    main()