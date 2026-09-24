import glob
import os

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    chunks = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        chunks.append({"id": doc_id, "text": text})
    return chunks

def build_index(persist_directory: str = CHROMA_DIR) -> chromadb.Collection:
    chunks = load_documents()
    print(f"Loaded {len(chunks)} document chunks from {DOCS_DIR}/")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (local, no API key)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    client = chromadb.PersistentClient(path=persist_directory)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": f"{c['id']}.txt"} for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks into ChromaDB collection '{COLLECTION_NAME}' at {persist_directory}/")
    return collection

if __name__ == "__main__":
    build_index()
