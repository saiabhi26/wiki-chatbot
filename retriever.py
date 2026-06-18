import json
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"  # Fast and effective
INDEX_PATH = "data/faiss_index.bin"
DOCS_PATH = "data/documents.json"
TEXTS_PATH = "data/texts.npy"

def build_index():
    """Encode all documents and save the FAISS index."""
    print("Loading documents...")
    with open(DOCS_PATH) as f:
        docs = json.load(f)

    texts = [d["text"][:512] for d in docs]  # Cap length for speed

    print("Encoding documents (this takes a while)...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    np.save(TEXTS_PATH, np.array(texts))

    print(f"✅ Index built with {index.ntotal} vectors")

def load_retriever():
    """Load model, index, and texts for querying."""
    model = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index(INDEX_PATH)
    texts = np.load(TEXTS_PATH, allow_pickle=True).tolist()
    return model, index, texts

def retrieve(query, model, index, texts, top_k=5):
    """Return top-k most relevant documents for a query."""
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [texts[i] for i in indices[0]]

if __name__ == "__main__":
    build_index()