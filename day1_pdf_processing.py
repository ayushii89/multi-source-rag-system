# day1_pdf_processing.py

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# ✅ Step 0: Clean text function
import re

def clean_text(text):
    # Remove weird characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Add space between letters and numbers
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    # Fix repeated words like linkedinlinkedin
    text = re.sub(r'\b(\w+)(\1)\b', r'\1', text)

    # Add space before URLs
    text = re.sub(r'(https?://)', r' \1', text)

    # Fix missing spaces before capital letters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# Step 1: Load PDF
def load_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # ✅ CLEAN TEXT HERE (VERY IMPORTANT)
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    return documents


# Step 2: Split text into chunks
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50
    )
    
    chunks = splitter.split_documents(documents)
    return chunks


# MAIN EXECUTION
if __name__ == "__main__":
    
    file_path = "sample.pdf"
    
    print("📄 Loading PDF...")
    docs = load_pdf(file_path)
    
    print(f"Total pages loaded: {len(docs)}")
    
    print("\n✂️ Splitting text...")
    chunks = split_text(docs)
    
    print(f"Total chunks created: {len(chunks)}")
    
    print("\n🔍 Sample chunk:")
    
    if chunks:
        print(chunks[0].page_content[:500])
    else:
        print("❌ No chunks created")