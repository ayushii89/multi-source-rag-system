# day6_ensemble_retriever.py

import os
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

load_dotenv()


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


# ── Step 1: Load Chroma ───────────────────────────────────────────────────────

def load_vector_store(embeddings, persist_dir: str = "chroma_index") -> Chroma:
    print(f"   Loading Chroma collection from '{persist_dir}'...")
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )


# ── Step 2: Build Ensemble Retriever (Semantic + BM25) ───────────────────────

def build_ensemble_retriever(db: Chroma):

    # Semantic retriever
    semantic_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Get all docs from Chroma for BM25
    all_docs = db.get()
    docs_for_bm25 = [
        Document(
            page_content=all_docs["documents"][i],
            metadata=all_docs["metadatas"][i]
        )
        for i in range(len(all_docs["documents"]))
    ]

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 4

    # Manual RRF fusion — no import needed
    def ensemble_retrieve(query: str) -> list:
        semantic_docs = semantic_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)

        seen = set()
        combined = []
        for doc in semantic_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)

        return combined[:6]

    print("   Ensemble retriever ready (Semantic + BM25)")
    return ensemble_retrieve


# ── Step 3: Format docs for prompt ───────────────────────────────────────────

def format_docs(docs: list) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


# ── Step 4: Build RAG chain ───────────────────────────────────────────────────

def build_rag_chain(llm, prompt):
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ── Step 5: Ask question + show citations ─────────────────────────────────────

def ask(chain, retriever, llm, prompt, query: str):
    print(f"\n🤔 Question: {query}")
    print("⏳ Thinking...\n")

    # Retrieve docs
    docs = retriever(query)
    context = format_docs(docs)

    # Run chain
    answer = chain.invoke({"context": context, "question": query})
    print(f"💬 Answer:\n{answer}")

    # Show sources
    print("\n📚 Sources used:")
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in seen:
            print(f"   → {source}")
            seen.add(source)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.environ["USER_AGENT"] = "rag-project/1.0"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("🧠 Loading vector store...")
    db = load_vector_store(embeddings)

    print("⚙️  Building ensemble retriever...")
    retriever = build_ensemble_retriever(db)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."
Always mention which source the answer came from.

Context:
{context}

Question: {question}

Answer:""")

    chain = build_rag_chain(llm, prompt)

    print("\n🔍 Ask anything from your PDF + Web + CSV sources!")
    print("Type 'quit' to exit\n")

    while True:
        query = input("Your question: ").strip()
        if query.lower() == "quit":
            break
        if query:
            ask(chain, retriever, llm, prompt, query)