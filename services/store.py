# services/store.py
import os
import uuid
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "rag-chunks"
DIM = 3072

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def get_index():
    indexes = pc.list_indexes().names()
    if INDEX_NAME not in indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # can be changed to match your OPENAI region but not required
            )
        )
    return pc.Index(INDEX_NAME)


def store_chunks(upload_id, upload_name, chunks, vectors):
    index = get_index()
    namespace = upload_id

    payloads = []
    for c, v in zip(chunks, vectors):
        pid = f"{upload_id}-{c['chunk_index']}"

        payloads.append({
            "id": pid,
            "values": v,
            "metadata": {
                "upload_id": upload_id,
                "upload_name": upload_name,
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "tokens": c["tokens"],
                "source_files": c["source_files"],
            }
        })

    index.upsert(vectors=payloads, namespace=namespace)
