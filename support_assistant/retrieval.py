import chromadb
from sentence_transformers import SentenceTransformer

from ingest import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

_model = None
_collection = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection

def retrieve_top_k(query: str, k: int = 3) -> list[dict]:

    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    hits = []
    for doc_id, text, distance in zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ):
        hits.append({"id": doc_id, "text": text, "distance": distance})
    return hits
