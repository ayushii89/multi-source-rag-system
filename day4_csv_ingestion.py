# day4_csv_ingestion.py

import re
import csv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


# ── Step 1: Load CSV and convert rows to natural language ─────────────────────

def load_csv(file_path: str) -> list:
    """
    Converts each CSV row into a natural language sentence.
    e.g. {"name": "Alice", "role": "Engineer", "salary": "90000"}
         → "name is Alice. role is Engineer. salary is 90000."
    """
    documents = []

    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            # Convert row to natural language
            sentence = ". ".join(
                f"{key.strip()} is {value.strip()}"
                for key, value in row.items()
                if value.strip()  # skip empty cells
            )
            sentence = clean_text(sentence)

            if len(sentence) > 30:  # skip near-empty rows
                doc = Document(
                    page_content=sentence,
                    metadata={
                        "source": file_path,
                        "row": i
                    }
                )
                documents.append(doc)

    print(f"   Rows loaded: {len(documents)}")
    return documents


# ── Step 2: Split (for large text cells) ─────────────────────────────────────

def split_text(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"   Chunks created: {len(chunks)}")
    return chunks


# ── Step 3: Add CSV chunks into existing Chroma collection ───────────────────

def update_vector_store(chunks: list, embeddings, persist_dir: str = "chroma_index") -> Chroma:
    print(f"   Loading existing Chroma collection from '{persist_dir}'...")

    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    print(f"   Adding {len(chunks)} CSV chunks...")
    db.add_documents(chunks)

    print(f"   Collection updated and saved.")
    return db


# ── Step 4: Search ────────────────────────────────────────────────────────────

def search(db: Chroma, query: str, k: int = 3) -> list:
    return db.similarity_search_with_score(query, k=k)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ["USER_AGENT"] = "rag-project/1.0"

    # ✏️ Point this to your CSV file
    csv_file = "sample.csv"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("📊 Loading CSV...")
    docs = load_csv(csv_file)

    print("✂️  Splitting...")
    chunks = split_text(docs)

    print("🧠 Updating Chroma collection...")
    db = update_vector_store(chunks, embeddings)

    print("\n🔍 Test your query across PDF + Web + CSV sources!")
    query = input("Enter your question: ")

    results = search(db, query)

    print("\n📌 Top results:\n")
    for i, (doc, score) in enumerate(results):
        print(f"--- Result {i+1}  (score: {score:.4f}) ---")
        print(doc.page_content[:400])
        print(f"📁 Source: {doc.metadata.get('source', '?')}  |  Row: {doc.metadata.get('row', '?')}\n")