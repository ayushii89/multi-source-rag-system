# day7_streamlit_app.py

import os
import re
import tempfile
import time
import numpy as np
from umap import UMAP
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import csv

load_dotenv()
os.environ["USER_AGENT"] = "rag-project/1.0"

st.set_page_config(
    page_title="NLP Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #f0ebff 0%, #e8e0ff 50%, #ede8ff 100%);
        min-height: 100vh;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1f5e 0%, #1e1442 100%);
        border-right: none;
    }

    [data-testid="stSidebar"] * {
        color: #e8e0ff !important;
    }

    .main-header {
        background: linear-gradient(135deg, #4a2c8a 0%, #6b3fa0 50%, #8b5cf6 100%);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
    }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        font-weight: 400;
        color: white;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-subtitle {
        font-family: 'DM Sans', sans-serif;
        color: rgba(255,255,255,0.7);
        font-size: 14px;
        margin-top: 8px;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 300;
    }

    .main-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin: 4px 4px 0 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }

    .user-bubble {
        background: linear-gradient(135deg, #6b3fa0 0%, #8b5cf6 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 20px 20px 4px 20px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(107, 63, 160, 0.3);
    }

    .bot-bubble {
        background: white;
        color: #2d1f5e;
        padding: 16px 20px;
        border-radius: 20px 20px 20px 4px;
        margin: 8px 0;
        max-width: 85%;
        border: 1px solid #e0d5ff;
        font-size: 15px;
        line-height: 1.7;
        box-shadow: 0 4px 15px rgba(107, 63, 160, 0.08);
    }

    .timestamp {
        font-size: 11px;
        color: #9c88c4;
        margin: 3px 6px;
    }

    .source-card {
        background: white;
        border: 1px solid #e0d5ff;
        border-radius: 12px;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 13px;
        color: #4a2c8a;
        box-shadow: 0 2px 8px rgba(107, 63, 160, 0.06);
    }

    .source-card a {
        color: #6b3fa0;
        text-decoration: none;
        font-weight: 500;
    }

    .stat-card {
        background: white;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        margin: 4px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }

    .stat-number {
        font-size: 26px;
        font-weight: 600;
        color: #8b5cf6;
        line-height: 1;
        font-family: 'DM Serif Display', serif;
    }

    .stat-label {
        font-size: 11px;
        color: #9c88c4;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .ingest-success {
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 10px;
        padding: 10px 14px;
        color: #c4b5fd;
        font-size: 13px;
        margin: 6px 0;
    }

    .sidebar-header {
        font-size: 10px;
        font-weight: 600;
        color: rgba(196, 181, 253, 0.7) !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 20px 0 8px 0;
    }

    .welcome-box {
        text-align: center;
        padding: 80px 20px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6b3fa0 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        padding: 8px 20px !important;
        box-shadow: 0 4px 12px rgba(107, 63, 160, 0.3) !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(107, 63, 160, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: white !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.4) !important;
    }

    .stChatInput > div {
        background: white !important;
        border: 2px solid #e0d5ff !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(107, 63, 160, 0.1) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 14px;
        padding: 4px;
        gap: 4px;
        box-shadow: 0 2px 10px rgba(107, 63, 160, 0.1);
        border: 1px solid #e0d5ff;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: #6b3fa0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        padding: 8px 20px !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6b3fa0, #8b5cf6) !important;
        color: white !important;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #6b3fa0, #8b5cf6) !important;
    }

    .stSpinner > div {
        border-top-color: #8b5cf6 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'\b(\w{4,})\1\b', r'\1', text)
    text = re.sub(r'(https?://)', r' \1', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_text(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)


def get_source_icon(source: str) -> str:
    if source.startswith("http"):
        return "🌐"
    elif source.endswith(".pdf"):
        return "📄"
    elif source.endswith(".csv"):
        return "📊"
    return "📁"


def clean_source(source: str) -> str:
    if not source:
        return "unknown"
    if source.startswith("http"):
        from urllib.parse import urlparse
        return urlparse(source).netloc
    if source.startswith("/var") or source.startswith("/tmp"):
        return source.split("/")[-1]
    return source


def format_source(source: str) -> str:
    if source.startswith("/var") or source.startswith("/tmp"):
        return source.split("/")[-1].replace(".pdf", " (PDF)")
    return source


def get_timestamp() -> str:
    return time.strftime("%H:%M")


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_pdf(uploaded_file) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name
    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source"] = uploaded_file.name
    docs = [d for d in docs if len(d.page_content.strip()) > 100]
    return split_text(docs)


def ingest_url(url: str) -> list:
    loader = WebBaseLoader([url])
    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    docs = [d for d in docs if len(d.page_content.strip()) > 100]
    return split_text(docs)


def ingest_csv(uploaded_file) -> list:
    content = uploaded_file.read().decode("utf-8").splitlines()
    reader = csv.DictReader(content)
    documents = []
    for i, row in enumerate(reader):
        sentence = ". ".join(
            f"{k.strip()} is {v.strip()}"
            for k, v in row.items() if v.strip()
        )
        sentence = clean_text(sentence)
        if len(sentence) > 30:
            documents.append(Document(
                page_content=sentence,
                metadata={"source": uploaded_file.name, "row": i}
            ))
    return split_text(documents)


# ── Vector store ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def get_vector_store(embeddings) -> Chroma:
    return Chroma(
        persist_directory="chroma_index",
        embedding_function=embeddings
    )


def build_ensemble_retriever(db: Chroma):
    semantic_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    all_docs = db.get()
    if not all_docs["documents"]:
        return None
    docs_for_bm25 = [
        Document(
            page_content=all_docs["documents"][i],
            metadata=all_docs["metadatas"][i]
        )
        for i in range(len(all_docs["documents"]))
    ]
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 6

    def ensemble_retrieve(query: str) -> list:
        semantic_docs = semantic_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)
        seen = set()
        combined = []
        for doc in semantic_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        return combined[:8]

    return ensemble_retrieve


def build_chain():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
    prompt = PromptTemplate.from_template("""
You are an expert research assistant. Use the context below to give a detailed, thorough answer.
- Write at least 3-5 sentences
- Use bullet points for lists or multiple points
- Always cite which source each piece of information came from
- If multiple sources mention the topic, combine their information
- Only use information from the context

Context:
{context}

Question: {question}

Detailed Answer:""")
    return prompt | llm | StrOutputParser()


def format_docs(docs: list) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


# ── Embedding visualization ───────────────────────────────────────────────────

def build_embedding_viz(db: Chroma):
    data = db.get(include=["documents", "metadatas", "embeddings"])
    if not data["documents"] or len(data["documents"]) < 5:
        return None

    embeddings_matrix = np.array(data["embeddings"])
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(15, len(data["documents"]) - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )
    coords = reducer.fit_transform(embeddings_matrix)

    sources = [clean_source(m.get("source", "unknown")) for m in data["metadatas"]]
    unique_sources = list(set(sources))
    colors = ["#8b5cf6", "#a78bfa", "#6b3fa0", "#c4b5fd", "#7c3aed", "#ddd6fe"]
    color_map = {src: colors[i % len(colors)] for i, src in enumerate(unique_sources)}

    fig = go.Figure()
    for source in unique_sources:
        indices = [i for i, s in enumerate(sources) if s == source]
        fig.add_trace(go.Scatter(
            x=[coords[i][0] for i in indices],
            y=[coords[i][1] for i in indices],
            mode="markers",
            name=source,
            marker=dict(
                size=9,
                color=color_map[source],
                opacity=0.85,
                line=dict(width=1, color="white")
            ),
            text=[data["documents"][i][:200] + "..." for i in indices],
            hovertemplate=f"<b>Source:</b> {source}<br><b>Chunk:</b><br>%{{text}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text="Embedding Space — Semantic clustering of your knowledge base",
            font=dict(size=15, color="#2d1f5e", family="DM Serif Display"),
            x=0.5
        ),
        paper_bgcolor="#f0ebff",
        plot_bgcolor="white",
        font=dict(color="#2d1f5e", family="DM Sans"),
        legend=dict(
            title="Sources",
            bgcolor="white",
            bordercolor="#e0d5ff",
            borderwidth=1,
            font=dict(color="#4a2c8a")
        ),
        xaxis=dict(showgrid=True, gridcolor="#e0d5ff", zeroline=False,
                   title="UMAP Dim 1", color="#6b3fa0"),
        yaxis=dict(showgrid=True, gridcolor="#e0d5ff", zeroline=False,
                   title="UMAP Dim 2", color="#6b3fa0"),
        height=550,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    return fig


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "ingested_sources" not in st.session_state:
    st.session_state.ingested_sources = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = "-"


# ── Load resources ────────────────────────────────────────────────────────────

embeddings = get_embeddings()
db = get_vector_store(embeddings)
chain = build_chain()

if st.session_state.retriever is None:
    st.session_state.retriever = build_ensemble_retriever(db)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 8px 0;">
        <div style="font-size: 2rem; margin-bottom: 8px;">🧠</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.3rem;
                    font-weight:400; color:white;">NLP Assistant</div>
        <div style="font-size:10px; color:rgba(196,181,253,0.7); margin-top:4px;
                    letter-spacing:2px; text-transform:uppercase;">
                    Knowledge · Retrieval · AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin:12px 0'>", unsafe_allow_html=True)

    st.markdown('<p class="sidebar-header">Session Stats</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.total_chunks}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{len(st.session_state.ingested_sources)}</div><div class="stat-label">Sources</div></div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.total_queries}</div><div class="stat-label">Queries</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{st.session_state.last_query_time}</div><div class="stat-label">Speed</div></div>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin:12px 0'>", unsafe_allow_html=True)

    st.markdown('<p class="sidebar-header">📄 PDF Document</p>', unsafe_allow_html=True)
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if pdf_file and st.button("⬆️ Ingest PDF", use_container_width=True):
        progress = st.progress(0, text="Reading PDF...")
        chunks = ingest_pdf(pdf_file)
        progress.progress(50, text="Embedding chunks...")
        db.add_documents(chunks)
        progress.progress(100, text="Done!")
        st.session_state.total_chunks += len(chunks)
        st.session_state.ingested_sources.append(("pdf", pdf_file.name))
        st.session_state.retriever = build_ensemble_retriever(db)
        progress.empty()
        st.markdown(f'<div class="ingest-success">✅ {len(chunks)} chunks from {pdf_file.name}</div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-header">🌐 Website</p>', unsafe_allow_html=True)
    url = st.text_input("Paste URL", placeholder="https://example.com", label_visibility="collapsed")
    if url and st.button("⬆️ Ingest URL", use_container_width=True):
        progress = st.progress(0, text="Scraping...")
        chunks = ingest_url(url)
        progress.progress(50, text="Embedding chunks...")
        db.add_documents(chunks)
        progress.progress(100, text="Done!")
        st.session_state.total_chunks += len(chunks)
        st.session_state.ingested_sources.append(("url", url))
        st.session_state.retriever = build_ensemble_retriever(db)
        progress.empty()
        st.markdown(f'<div class="ingest-success">✅ {len(chunks)} chunks from {url[:40]}...</div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-header">📊 CSV File</p>', unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if csv_file and st.button("⬆️ Ingest CSV", use_container_width=True):
        progress = st.progress(0, text="Processing CSV...")
        chunks = ingest_csv(csv_file)
        progress.progress(50, text="Embedding chunks...")
        db.add_documents(chunks)
        progress.progress(100, text="Done!")
        st.session_state.total_chunks += len(chunks)
        st.session_state.ingested_sources.append(("csv", csv_file.name))
        st.session_state.retriever = build_ensemble_retriever(db)
        progress.empty()
        st.markdown(f'<div class="ingest-success">✅ {len(chunks)} chunks from {csv_file.name}</div>', unsafe_allow_html=True)

    if st.session_state.ingested_sources:
        st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin:12px 0'>", unsafe_allow_html=True)
        st.markdown('<p class="sidebar-header">✅ Ingested Sources</p>', unsafe_allow_html=True)
        for kind, source in st.session_state.ingested_sources:
            icon = "📄" if kind == "pdf" else "🌐" if kind == "url" else "📊"
            label = source if len(source) < 32 else source[:32] + "..."
            st.markdown(f'<div class="ingest-success">{icon} {label}</div>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin:12px 0'>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <div class="main-title">🧠 NLP Knowledge Assistant</div>
    <div class="main-subtitle">Semantic Search · Hybrid Retrieval · Generative AI</div>
    <div style="margin-top: 12px;">
        <span class="main-badge">RAG Pipeline</span>
        <span class="main-badge">BM25 + Semantic</span>
        <span class="main-badge">Llama 3.1</span>
        <span class="main-badge">UMAP Visualization</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬 Chat", "🔭 Embedding Space"])


# ══════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════

with tab1:
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-box">
            <div style="font-size:56px; margin-bottom:20px;">✨</div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.6rem;
                        color:#2d1f5e; font-weight:400; margin-bottom:12px;">
                Your intelligent knowledge companion
            </div>
            <div style="font-size:15px; color:#6b3fa0; margin-bottom:8px;">
                Upload documents · Ask questions · Get cited answers
            </div>
            <div style="display:flex; justify-content:center; gap:12px; margin-top:24px; flex-wrap:wrap;">
                <div style="background:white; border:1px solid #e0d5ff; border-radius:12px;
                            padding:12px 20px; font-size:13px; color:#4a2c8a;
                            box-shadow:0 2px 8px rgba(107,63,160,0.08);">
                    📄 Upload any PDF
                </div>
                <div style="background:white; border:1px solid #e0d5ff; border-radius:12px;
                            padding:12px 20px; font-size:13px; color:#4a2c8a;
                            box-shadow:0 2px 8px rgba(107,63,160,0.08);">
                    🌐 Scrape any website
                </div>
                <div style="background:white; border:1px solid #e0d5ff; border-radius:12px;
                            padding:12px 20px; font-size:13px; color:#4a2c8a;
                            box-shadow:0 2px 8px rgba(107,63,160,0.08);">
                    📊 Analyze any CSV
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end; margin: 12px 0;">
                <div>
                    <div class="user-bubble">{msg["content"]}</div>
                    <div class="timestamp" style="text-align:right">{msg.get("time","")}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-start; margin: 12px 0;">
                <div style="width:85%">
                    <div class="bot-bubble">{msg["content"]}</div>
                    <div class="timestamp">{msg.get("time","")}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            if msg.get("sources"):
                with st.expander("📚 Sources used", expanded=False):
                    for source in msg["sources"]:
                        icon = get_source_icon(source)
                        if source.startswith("http"):
                            st.markdown(f'<div class="source-card">{icon} <a href="{source}" target="_blank">{source}</a></div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="source-card">{icon} {format_source(source)}</div>', unsafe_allow_html=True)

    if query := st.chat_input("Ask anything from your knowledge base..."):
        ts = get_timestamp()
        st.session_state.messages.append({"role": "user", "content": query, "time": ts})
        st.session_state.total_queries += 1

        if st.session_state.retriever is None:
            st.warning("⚠️ Please ingest at least one source first.")
        else:
            start = time.time()
            with st.spinner("Searching knowledge base and generating answer..."):
                docs = st.session_state.retriever(query)
                context = format_docs(docs)
                answer = chain.invoke({"context": context, "question": query})

            elapsed = round(time.time() - start, 1)
            st.session_state.last_query_time = f"{elapsed}s"
            sources = list({doc.metadata.get("source", "unknown") for doc in docs})
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "time": get_timestamp()
            })

        st.rerun()


# ══════════════════════════════════════════════
# TAB 2 — EMBEDDING VISUALIZATION
# ══════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <div style="font-family:'DM Serif Display',serif; font-size:1.4rem;
                    color:#2d1f5e; margin-bottom:4px;">
            🔭 Embedding Space Visualization
        </div>
        <div style="font-size:13px; color:#6b3fa0;">
            Each dot represents a text chunk · Proximity encodes semantic similarity · Color indicates source
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Generate Plot", use_container_width=True):
            with st.spinner("Running UMAP dimensionality reduction..."):
                fig = build_embedding_viz(db)
                if fig:
                    st.session_state["viz_fig"] = fig
                else:
                    st.warning("Ingest at least 5 chunks first.")

    with col2:
        st.markdown("""
        <div style="background:white; border:1px solid #e0d5ff; border-radius:14px;
                    padding:14px 18px; font-size:13px; color:#4a2c8a;
                    box-shadow:0 2px 8px rgba(107,63,160,0.08);">
            💡 <b>How to read this chart:</b> Chunks from the same source naturally cluster together.
            Semantically similar content from different sources appears nearby.
            Hover over any dot to preview the chunk text.
        </div>
        """, unsafe_allow_html=True)

    if "viz_fig" in st.session_state:
        st.plotly_chart(st.session_state["viz_fig"], use_container_width=True)