# embedding/preview_embedding.py
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "text-embedding-3-large"

def embed_sentences(sentences):
    inputs = [f"passage: {s}" for s in sentences]
    resp = client.embeddings.create(model=MODEL, input=inputs)
    return [v.embedding for v in resp.data]
