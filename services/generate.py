# services/generate.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GEN_MODEL = "gpt-4o" 

def generate_answer(query, ranked_chunks, top_k=8):
    """
    ranked_chunks = reranked (LLM) chunks sorted best→worst
    """
    if not ranked_chunks:
        return {
            "answer": "I don't know based on the provided context.",
            "citations": [],
            "confidence": 0.0
        }

    selected = ranked_chunks[:top_k]

    context = "\n\n".join(
        f"[{c['chunk_index']}] {c['text']}" for c in selected
    )

    prompt = f"""
You are a precision-focused RAG system.
Your goal is to answer the user's question using only the provided context chunks.

Instructions:
1. Analyze the Context carefully.
2. If the answer is not explicitly in the context, state: "I don't know based on the provided context."
3. Do not make up information.

QUERY:
{query}

CONTEXT:
{context}

Output Format:
Answer the question directly and concisely.
"""

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    answer = resp.choices[0].message.content.strip()

    conf = 0.0
    if selected:
        conf = sum(c.get("rerank_score", 0) for c in selected) / len(selected)

    citations = [c["chunk_index"] for c in selected]

    return {
        "answer": answer,
        "citations": citations,
        "confidence": round(conf, 3)
    }