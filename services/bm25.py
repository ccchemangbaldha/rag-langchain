# services/bm25.py
from rank_bm25 import BM25Okapi

def bm25_search(query, chunks, top_k=8):
    if not chunks:
        return []

    corpus = [c["text"] for c in chunks]
    tokenized = [d.split() for d in corpus]
    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores(query.split())
    ranked = list(zip(chunks, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    out = []
    for c, s in ranked[:top_k]:
        c["bm25_score"] = float(s)
        out.append(c)

    return out
