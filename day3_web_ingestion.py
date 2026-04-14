# day3_web_ingestion.py

import re
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ── Same clean_text from Day 1 ────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'\b(\w{4,})\1\b', r'\1', text)
    text = re.sub(r'(https?://)', r' \1', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Step 1: Load websites ─────────────────────────────────────────────────────

def load_websites(urls: list) -> list:
    loader = WebBaseLoader(urls)
    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    docs = [d for d in docs if len(d.page_content.strip()) > 100]

    print(f"   Pages loaded: {len(docs)}")
    return docs


# ── Step 2: Split ─────────────────────────────────────────────────────────────

def split_text(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"   Chunks created: {len(chunks)}")
    return chunks


# ── Step 3: Load existing Chroma collection and add web chunks ────────────────

def update_vector_store(chunks: list, embeddings, persist_dir: str = "chroma_index") -> Chroma:
    print(f"   Loading existing Chroma collection from '{persist_dir}'...")

    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    print(f"   Adding {len(chunks)} web chunks...")
    db.add_documents(chunks)

    print(f"   Collection updated and saved.")
    return db


# ── Step 4: Search ────────────────────────────────────────────────────────────

def search(db: Chroma, query: str, k: int = 3) -> list:
    return db.similarity_search_with_score(query, k=k)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ✏️ Swap these URLs for whatever topic your PDF covers
    urls = [
        "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        "https://python.langchain.com/docs/introduction/",
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("🌐 Loading websites...")
    docs = load_websites(urls)

    print("✂️  Splitting...")
    chunks = split_text(docs)

    print("🧠 Updating Chroma collection...")
    db = update_vector_store(chunks, embeddings)

    print("\n🔍 Test your query across PDF + Web sources!")
    query = input("Enter your question: ")

    results = search(db, query)

    print("\n📌 Top results:\n")
    for i, (doc, score) in enumerate(results):
        print(f"--- Result {i+1}  (score: {score:.4f}) ---")
        print(doc.page_content[:400])
        print(f"🔗 Source: {doc.metadata.get('source', '?')}\n")