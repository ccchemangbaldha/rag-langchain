# services/retrieve_chunks.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

load_dotenv()

INDEX_NAME = "rag-chunks"
MODEL = "text-embedding-3-large"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_query(q):
    r = openai_client.embeddings.create(
        model=MODEL,
        input=q
    )
    return r.data[0].embedding


def retrieve_chunks(query, upload_id, limit=5, threshold=0.0):
    # ensure index exists
    indexes = pc.list_indexes().names()
    if INDEX_NAME not in indexes:
        return []   # nothing stored yet

    index = pc.Index(INDEX_NAME)

    vec = embed_query(query)

    result = index.query(
        vector=vec,
        namespace=upload_id,
        top_k=limit,
        include_metadata=True
    )

    out = []
    for m in result.matches:
        if m.score >= threshold:
            out.append({
                "score": round(m.score, 4),
                "chunk_index": m.metadata.get("chunk_index"),
                "text": m.metadata.get("text"),
                "tokens": m.metadata.get("tokens"),
                "source_files": m.metadata.get("source_files"),
                "upload_id": m.metadata.get("upload_id")
            })

    return out
