# day8_embedding_viz.py

import os
import numpy as np
import plotly.graph_objects as go
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from umap import UMAP
from sklearn.preprocessing import LabelEncoder

os.environ["USER_AGENT"] = "rag-project/1.0"


# ── Load Chroma ───────────────────────────────────────────────────────────────

def load_data(persist_dir: str = "chroma_index"):
    print("🧠 Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("📦 Loading Chroma collection...")
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    data = db.get(include=["documents", "metadatas", "embeddings"])
    return data


# ── Reduce dimensions with UMAP ───────────────────────────────────────────────

def reduce_dimensions(embeddings_matrix: np.ndarray) -> np.ndarray:
    print("🔭 Running UMAP dimensionality reduction...")
    reducer = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )
    return reducer.fit_transform(embeddings_matrix)


# ── Clean source name ─────────────────────────────────────────────────────────

def clean_source(source: str) -> str:
    if not source:
        return "unknown"
    if source.startswith("http"):
        # shorten URL to domain
        try:
            from urllib.parse import urlparse
            return urlparse(source).netloc
        except:
            return source[:40]
    if source.startswith("/var") or source.startswith("/tmp"):
        return source.split("/")[-1]
    return source


# ── Build Plotly visualization ────────────────────────────────────────────────

def build_viz(data, coords: np.ndarray):
    print("🎨 Building visualization...")

    documents = data["documents"]
    metadatas = data["metadatas"]

    # Group by source
    sources = [clean_source(m.get("source", "unknown")) for m in metadatas]
    unique_sources = list(set(sources))

    # Color palette
    colors = [
        "#667eea", "#f093fb", "#4facfe",
        "#43e97b", "#fa709a", "#fee140",
        "#a18cd1", "#fda085", "#84fab0"
    ]
    color_map = {src: colors[i % len(colors)] for i, src in enumerate(unique_sources)}

    fig = go.Figure()

    for source in unique_sources:
        indices = [i for i, s in enumerate(sources) if s == source]
        x = [coords[i][0] for i in indices]
        y = [coords[i][1] for i in indices]

        # Truncate text for hover
        texts = [
            documents[i][:200].replace("<", "&lt;").replace(">", "&gt;") + "..."
            for i in indices
        ]

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=source,
            marker=dict(
                size=8,
                color=color_map[source],
                opacity=0.8,
                line=dict(width=0.5, color="white")
            ),
            text=texts,
            hovertemplate=(
                f"<b>Source:</b> {source}<br>"
                "<b>Chunk:</b><br>%{text}<extra></extra>"
            )
        ))

    fig.update_layout(
        title=dict(
            text="🧠 Embedding Space — Multi-Source RAG",
            font=dict(size=20, color="white"),
            x=0.5
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1e2130",
        font=dict(color="white"),
        legend=dict(
            title="Sources",
            bgcolor="#1e2130",
            bordercolor="#2d2f3e",
            borderwidth=1,
            font=dict(color="white")
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#2d2f3e",
            zeroline=False,
            title="UMAP Dimension 1"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#2d2f3e",
            zeroline=False,
            title="UMAP Dimension 2"
        ),
        width=1000,
        height=650,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_data()

    if not data["documents"]:
        print("❌ No documents found in Chroma index. Run day2-day4 first.")
        exit()

    print(f"   Total chunks loaded: {len(data['documents'])}")

    # Convert embeddings to numpy array
    embeddings_matrix = np.array(data["embeddings"])
    print(f"   Embedding shape: {embeddings_matrix.shape}")

    # Reduce to 2D
    coords = reduce_dimensions(embeddings_matrix)

    # Build and show plot
    fig = build_viz(data, coords)

    # Save as HTML
    output_path = "embedding_viz.html"
    fig.write_html(output_path)
    print(f"\n✅ Visualization saved to '{output_path}'")
    print("   Opening in browser...")

    fig.show()