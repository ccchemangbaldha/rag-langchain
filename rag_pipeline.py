# rag_pipeline.py
import uuid
import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from qdrant_client import models, QdrantClient
import spacy
import hdbscan

from parser.file_intake import parse_file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

EMBED_MODEL = "text-embedding-3-large"
nlp = spacy.load("en_core_web_sm")

client_openai = OpenAI(api_key=OPENAI_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_ENDPOINT,
    api_key=QDRANT_API_KEY
)

COL = "first-col"
DIM = 3072

def ensure_collection():
    info = qdrant.get_collections()
    names = [c.name for c in info.collections]
    if COL not in names:
        qdrant.create_collection(
            collection_name=COL,
            vectors_config={
                "first-dense-vector": models.VectorParams(
                    size=DIM,
                    distance=models.Distance.COSINE
                )
            }
        )

def embed_sentences(sentences):
    inputs = [f"passage: {s}" for s in sentences]
    resp = client_openai.embeddings.create(model=EMBED_MODEL, input=inputs)
    return [v.embedding for v in resp.data]

def cluster_embeddings(vectors):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(vectors)
    return labels

def process_files(paths):
    all_contents = []
    filenames = []

    for p in paths:
        items = parse_file(p)
        for it in items:
            if it.get("content"):
                all_contents.append(it["content"])
                filenames.append(it["filename"])

    merged = "\n".join(all_contents).strip()
    if not merged:
        return

    uid = str(uuid.uuid4())
    short_title = " ".join(merged.split()[:12])
    upload_name = f"{short_title}-{uid}"

    doc = nlp(merged)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    vectors = embed_sentences(sentences)
    labels = cluster_embeddings(vectors)

    ensure_collection()

    points = []
    for idx, (sent, vec) in enumerate(zip(sentences, vectors)):
        payload = {
            "upload_id": uid,
            "upload_name": upload_name,
            "sentence": sent,
            "cluster": int(labels[idx]),
            "sentence_index": idx,
            "source_files": filenames,
            "language": "en",
            "tokens": len(sent.split())
        }

        points.append(
            models.PointStruct(
                id=f"{uid}-{idx}",
                vector={"first-dense-vector": vec},
                payload=payload,
            )
        )

    qdrant.upsert(collection_name=COL, points=points)

    return {"upload_name": upload_name, "uuid": uid, "sentences": len(sentences)}
