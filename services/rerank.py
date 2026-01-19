# services/rerank.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RERANK_MODEL = "gpt-4o-mini"

def rerank(query, chunks):
    ranked = []

    for c in chunks:
        prompt = f"""
Query: {query}
Text: {c['text']}

On a scale of 0.0 to 1.0, how relevant is this text to the query?
Output ONLY the number.
"""
        try:
            resp = client.chat.completions.create(
                model=RERANK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            score_str = resp.choices[0].message.content.strip()
            score = float(score_str)
        except:
            score = 0.0
            
        c["rerank_score"] = score
        ranked.append(c)

    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return ranked