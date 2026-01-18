# services/retrieve_chunks.py
import os
from openai import OpenAI
from qdrant_client import QdrantClient, models

QDRANT_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COL = "first-col"
VECTOR_FIELD = "first-dense-vector"

client = QdrantClient(
    url=QDRANT_ENDPOINT,
    api_key=QDRANT_API_KEY
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def embed_query(query: str):
    resp = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=f"query: {query}"
    )
    return resp.data[0].embedding

def retrieve_chunks(query: str, limit=10, threshold=0.0):
    qvec = embed_query(query)

    results = client.search(
        collection_name=COL,
        query_vector={VECTOR_FIELD: qvec},
        limit=limit
    )

    filtered = [
        {
            "score": round(r.score, 4),
            "text": r.payload.get("text"),
            "chunk_index": r.payload.get("chunk_index"),
            "upload_id": r.payload.get("upload_id"),
            "source_files": r.payload.get("source_files"),
            "tokens": r.payload.get("tokens")
        }
        for r in results if r.score >= threshold
    ]

    return filtered
