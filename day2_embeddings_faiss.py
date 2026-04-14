# day2_embeddings_faiss.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# Step 2: Split text
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

# Step 3: Create embeddings + FAISS
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
    db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_index")

    return db

# Step 4: Search
def search_query(db, query):
    results = db.similarity_search(query, k=3)
    return results


# MAIN
if __name__ == "__main__":

    file_path = "sample.pdf"

    print("📄 Loading PDF...")
    docs = load_pdf(file_path)

    print("✂️ Splitting...")
    chunks = split_text(docs)

    print("🧠 Creating embeddings + FAISS...")
    db = create_vector_store(chunks)

    print("🔍 Ask something from your PDF!")
    query = input("Enter your question: ")

    results = search_query(db, query)

    print("\n📌 Top relevant chunks:\n")

    for i, res in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(res.page_content[:300])