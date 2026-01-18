# services/store.py
import uuid
import os
from qdrant_client import QdrantClient, models

QDRANT_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COL = "first-col"
VECTOR_FIELD = "first-dense-vector"

client = QdrantClient(
    url=QDRANT_ENDPOINT,
    api_key=QDRANT_API_KEY
)

def store_chunks(upload_id, upload_name, chunks, vectors):
    points = []

    for c, v in zip(chunks, vectors):
        composite_id = f"{upload_id}-{c['chunk_index']}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, composite_id))

        payload = {
            "custom_id": composite_id,
            "upload_id": upload_id,
            "upload_name": upload_name,
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            "tokens": c["tokens"],
            "source_files": c["source_files"],
            "language": "en"
        }

        points.append(
            models.PointStruct(
                id=point_id,
                vector={VECTOR_FIELD: v},
                payload=payload
            )
        )

    client.upsert(collection_name=COL, points=points)
