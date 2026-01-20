import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GEN_MODEL = "gpt-4o"
IMAGE_MODEL = "dall-e-3"

def generate_image(query):
    """
    Generates a simple educational illustration.
    """
    try:
        image_prompt = (
            f"A simple, clean, educational illustration explaining: {query}. "
            f"Cartoon style or textbook diagram style. "
            f"Bright colors, white background, easy to understand."
        )
        
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
            "answer": "I don't know based on the provided content. 😕",
            "citations": [],
            "confidence": 0.0,
            "image_url": None
        }

    selected = ranked_chunks[:top_k]
    context = "\n\n".join(f"[{c['chunk_index']}] {c['text']}" for c in selected)

    # Student-Friendly Prompt
    prompt = f"""
You are a friendly and intelligent Tutor AI.
Your goal is to answer the student's question CLEARLY and CONCISELY using ONLY the context below.

Instructions:
1. Use the provided Context to answer.
2. If the answer is NOT in the context, you MUST say exactly: "I don't know based on the provided content."
3. Do not make up facts.
4. Use formatting like bullet points, bold text, emojis and paragraphs to make it easy to read.
5. Be encouraging but factual.

QUESTION:
{query}

CONTEXT:
{context}
"""

    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    answer = resp.choices[0].message.content.strip()

    negative_phrases = ["I don't know based on the provided content", "I don't know"]
    
    is_negative_answer = any(phrase in answer for phrase in negative_phrases)
    
    image_url = None
    if create_visual and not is_negative_answer:
        image_url = generate_image(query)

    conf = sum(c.get("rerank_score", 0) for c in selected) / len(selected) if selected else 0
    citations = [c["chunk_index"] for c in selected]

    return {
        "answer": answer,
        "citations": citations,
        "confidence": round(conf, 3),
        "image_url": image_url
    }