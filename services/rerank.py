# services/rerank.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
RERANK_MODEL = "gpt-4.1-mini"

def rerank(query, chunks):
    ranked = []
    for c in chunks:
        prompt = f"""
Query: {query}
Chunk: {c['text']}

Rate usefulness 0.0-1.0:
Only number.
"""
        resp = client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        score = float(resp.choices[0].message.content.strip())
        c["rerank_score"] = score
        ranked.append(c)

    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked
