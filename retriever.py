import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Chunking config: word-based, with overlap so context isn't lost at
# chunk boundaries. Replaces the old approach of truncating each
# document to its first 512 characters (which discarded everything
# past that point and could cut off mid-sentence).
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30

EMBED_MODEL = "all-MiniLM-L6-v2" 
INDEX_PATH = "data/faiss_index.bin"
DOCS_PATH = "data/documents.json"
TEXTS_PATH = "data/texts.npy"


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks

def build_index():
    """Chunk, Encode all documents and save the FAISS index."""
    print("Loading documents...")
    with open(DOCS_PATH) as f:
        docs = json.load(f)

    print("Chunking documents...")
    texts = []
    for d in docs:
        texts.extend(chunk_text(d["text"]))
    print(f"  → {len(docs)} documents split into {len(texts)} chunks")

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
    """Return top-k most relevant documents and their distances for a query.
     Lower distance = more relevant (this is L2 distance, not similarity).
    """
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    results = [texts[i] for i in indices[0]]
    return results, distances[0]

if __name__ == "__main__":
    build_index()
