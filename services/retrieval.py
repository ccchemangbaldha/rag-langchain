# services/retrieval.py
import os
from qdrant_client import QdrantClient, models
from openai import OpenAI

QDRANT_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = QdrantClient(
    url=QDRANT_ENDPOINT,
    api_key=QDRANT_API_KEY
)

COL = "first-col"
DENSE = "first-dense-vector"
SPARSE = "first-sparse-vector"

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def embed_query(query: str):
    resp = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=f"query: {query}"
    )
    return resp.data[0].embedding

def retrieve(query: str, top_k=5):
    dense = embed_query(query)

    # sparse comes from keyword tokenization (BM25-ish)
    sparse = models.SparseVector(
        indices=[i for i, _ in enumerate(query.split())],
        values=[1.0 for _ in query.split()]
    )

    result = client.search(
        collection_name=COL,
        query_vector={
            DENSE: dense,
            SPARSE: sparse
        },
        limit=top_k
    )

    return [
        {
            "score": r.score,
            "text": r.payload.get("text"),
            "chunk_index": r.payload.get("chunk_index"),
            "upload_id": r.payload.get("upload_id"),
            "source_files": r.payload.get("source_files"),
            "tokens": r.payload.get("tokens")
        }
        for r in result
    ]
