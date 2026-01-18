# services/hybrid.py

from services.bm25 import bm25_search
from services.rrf import rrf_fuse
from services.rerank import rerank
from services.generate import generate_answer

def hybrid_rag(query, dense_chunks, sparse_top_k=10, final_top_k=5):
    if not dense_chunks:
        return {
            "answer": "I don't know based on the provided context.",
            "citations": [],
            "confidence": 0.0
        }

    # sparse from BM25
    sparse = bm25_search(query, dense_chunks, top_k=sparse_top_k)

    # fuse → produces ordered IDs
    fused = rrf_fuse(dense_chunks, sparse)

    # safe lookup
    idx_lookup = {c["chunk_index"]: c for c in dense_chunks + sparse}
    fused_chunks = [idx_lookup[c["chunk_index"]] for c in fused]  # fused returns chunk objects now

    # rerank via LLM
    reranked = rerank(query, fused_chunks)

    # select final evidence
    final_chunks = reranked[:final_top_k]

    return generate_answer(query, final_chunks)
