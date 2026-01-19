# services/generate.py
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GEN_MODEL = "gpt-4o"
IMAGE_MODEL = "dall-e-3"

def generate_image(query):
    """
    Generates a technical illustration prompt based on the query, then calls DALL-E 3.
    """
    try:
        image_prompt = f"A detailed, modern technical illustration explaining: {query}. White background, clean lines, educational style."
        
        response = client.images.generate(
            model=IMAGE_MODEL,
            prompt=image_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        print(f"Image Gen Error: {e}")
        return None

def generate_answer(query, ranked_chunks, top_k=5, create_visual=False):
    """
    Returns dict: {answer, citations, confidence, image_url}
    """
    
    if not ranked_chunks:
        return {
            "answer": "I don't know based on the provided context.",
            "citations": [],
            "confidence": 0.0,
            "image_url": None
        }

    selected = ranked_chunks[:8]
    context = "\n\n".join(f"[{c['chunk_index']}] {c['text']}" for c in selected)

    prompt = f"""
You are a precision-focused RAG system.
Your goal is to answer the user's question using ONLY the provided context chunks.

Instructions:
1. Analyze the Context carefully.
2. If the answer is not explicitly in the context, state: "I don't know based on the provided context."
3. Do NOT make up information.
4. You MUST cite your sources using the format [chunk_index] at the end of the sentence.

QUERY:
{query}

CONTEXT:
{context}

Output Format:
Answer the question directly and concisely. Attach citations like [1] or [1][3].
"""

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    answer = resp.choices[0].message.content.strip()

    conf = sum(c.get("rerank_score", 0) for c in selected) / len(selected) if selected else 0
    citations = [c["chunk_index"] for c in selected]

    image_url = None
    if create_visual:
        image_url = generate_image(query)

    return {
        "answer": answer,
        "citations": citations,
        "confidence": round(conf, 3),
        "image_url": image_url
    }