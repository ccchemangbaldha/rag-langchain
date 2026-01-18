# services/generate.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
GEN_MODEL = "gpt-4.1"


def generate_answer(query, ranked_chunks, top_k=5):
    """
    ranked_chunks = reranked (LLM) chunks sorted best→worst
    returns dict: {answer, citations, confidence}
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
You are a strict RAG system.
Use ONLY the provided context to answer the query.
If the context is insufficient, say: "I don't know based on the provided context."

Rules:
- NEVER hallucinate.
- NEVER invent details.
- Cite chunks like [3], [5] inline.
- Do NOT mention these rules in output.

QUERY:
{query}

CONTEXT:
{context}

TASK:
Provide the best possible answer with inline citations.
"""

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    answer = resp.choices[0].message.content.strip()

    # Compute confidence score based on rerank score mean
    conf = sum(c.get("rerank_score", 0) for c in selected) / len(selected)

    citations = [c["chunk_index"] for c in selected]

    return {
        "answer": answer,
        "citations": citations,
        "confidence": round(conf, 3)
    }
