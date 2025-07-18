import os
import fitz  # PyMuPDF
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# === CONFIG ===
PDF_PATH = "./company_docs/KGT Solutions - Company Data.pdf"
INDEX_PATH = "./company_docs/index.faiss"
CHUNKS_PATH = "./company_docs/chunks.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = 'all-MiniLM-L6-v2'

# === Step 1: Extract Text from PDF ===
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

# === Step 2: Split Text into Overlapping Chunks ===
def split_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# === Step 3: Embed Chunks ===
def embed_chunks(chunks, model):
    return model.encode(chunks, show_progress_bar=True)

# === Step 4: Build and Save FAISS Index ===
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# === Main Pipeline ===
def main():
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(PDF_PATH)

    print("Splitting text into chunks...")
    chunks = split_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"Embedding {len(chunks)} chunks...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_chunks(chunks, model)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"Saving FAISS index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)

    print(f"Saving chunks to {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Indexing complete.")

if __name__ == "__main__":
    os.makedirs("./company_docs", exist_ok=True)
    main()
