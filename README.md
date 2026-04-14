cat > README.md << 'EOF'
# 🧠 Multi-Source RAG System

An advanced Retrieval-Augmented Generation (RAG) system that ingests data from multiple sources and answers natural language questions with source citations.

## 🚀 Features

- **Multi-source ingestion** — PDFs, websites, and CSV files
- **Hybrid retrieval** — Semantic search + BM25 keyword search + Sentence Window retrieval
- **Ensemble fusion** — Reciprocal Rank Fusion (RRF) combining all retrievers
- **LLM generation** — Llama 3.1 via Groq API with source citations
- **UMAP visualization** — Interactive embedding space visualization
- **Streamlit UI** — Clean web interface with real-time ingestion and chat

## 🏗️ Architecture
User Query
↓
Streamlit UI
↓
Data Ingestion (PDF + Web + CSV)
↓
HuggingFace Embeddings (all-MiniLM-L6-v2)
↓
Chroma Vector Store
↓
Multi-Retriever System
├── Semantic Search (cosine similarity)
├── BM25 Keyword Search
└── Sentence Window Retrieval
↓
RRF Fusion + Re-ranking
↓
Llama 3.1 (Groq API)
↓
Answer + Source Citations

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | Chroma |
| LLM | Llama 3.1 via Groq API |
| Retrieval | LangChain + LlamaIndex |
| Keyword Search | BM25 |
| Visualization | UMAP + Plotly |
| Frontend | Streamlit |
| PDF Loader | PyMuPDF |

## 📁 Project Structure
rag-project/
├── day1_pdf_processing.py      # PDF loading + text cleaning
├── day2_embeddings_faiss.py    # Embeddings + Chroma vector store
├── day3_web_ingestion.py       # Website scraping + ingestion
├── day4_csv_ingestion.py       # CSV to natural language + ingestion
├── day5_rag_chain.py           # LLM + RAG chain with Groq
├── day6_ensemble_retriever.py  # BM25 + semantic ensemble retrieval
├── day7_streamlit_app.py       # Full Streamlit UI
├── day8_embedding_viz.py       # UMAP embedding visualization
├── day9_sentence_window.py     # Sentence window retrieval
└── .env                        # API keys (not committed)
## ⚙️ Setup

**1. Clone the repo:**
```bash
git clone https://github.com/ayushii89/multi-source-rag-system.git
cd multi-source-rag-system
```

**2. Install dependencies:**
```bash
pip install langchain langchain-community langchain-chroma langchain-groq
pip install langchain-huggingface langchain-text-splitters
pip install streamlit umap-learn plotly rank-bm25
pip install pymupdf python-dotenv llama-index
pip install llama-index-embeddings-huggingface
```

**3. Set up environment variables:**
```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

Get your free Groq API key at: https://console.groq.com

**4. Run the app:**
```bash
streamlit run day7_streamlit_app.py
```

## 🎯 Usage

1. Upload a PDF document using the sidebar
2. Paste a website URL and click Ingest URL
3. Upload a CSV file
4. Ask questions in the chat interface
5. View source citations for every answer
6. Explore the Embedding Space tab to visualize semantic clustering

## 📊 Results

- Multi-source retrieval working across PDF, web, and CSV simultaneously
- Source citations shown for every answer
- UMAP visualization showing semantic clustering by source type
- Response time: 1-3 seconds per query

## 👩‍💻 Author

Built by Ayushee 
