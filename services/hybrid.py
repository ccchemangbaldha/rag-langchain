# services/hybrid.py

from services.bm25 import bm25_search
from services.rrf import rrf_fuse
from services.rerank import rerank
from services.generate import generate_answer

def hybrid_rag(query, dense_chunks, sparse_top_k=10, final_top_k=5, enable_image=False):
    """
    Added 'enable_image' parameter to control DALL-E generation.
    """
    if not dense_chunks:
        return {
            "answer": "I don't know based on the provided context.",
            "citations": [],
            "confidence": 0.0,
            "image_url": None
        }

    sparse = bm25_search(query, dense_chunks, top_k=sparse_top_k)

    fused = rrf_fuse(dense_chunks, sparse)

    idx_lookup = {c["chunk_index"]: c for c in dense_chunks + sparse}
    fused_chunks = [idx_lookup[c["chunk_index"]] for c in fused]

    reranked = rerank(query, fused_chunks)
    final_chunks = reranked[:final_top_k]

    return generate_answer(query, final_chunks, create_visual=enable_image)