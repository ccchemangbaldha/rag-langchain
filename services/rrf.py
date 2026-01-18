# services/rrf.py

def rrf_fuse(dense_chunks, sparse_chunks, k=60):
    """
    dense_chunks  = list of chunks sorted semantically (pinecone)
    sparse_chunks = list of chunks sorted lexically (BM25)
    returns fused ordered chunks (best first)
    """

    if not dense_chunks and not sparse_chunks:
        return []

    dense_rank = {c["chunk_index"]: i for i, c in enumerate(dense_chunks)}
    sparse_rank = {c["chunk_index"]: i for i, c in enumerate(sparse_chunks)}

    # all participating chunk IDs
    all_ids = set(dense_rank.keys()) | set(sparse_rank.keys())

    fused = []
    for cid in all_ids:
        dr = dense_rank.get(cid, 9999)
        sr = sparse_rank.get(cid, 9999)
        score = (1 / (k + dr)) + (1 / (k + sr))
        fused.append((cid, score))

    fused.sort(key=lambda x: x[1], reverse=True)

    # reconstruct full chunk objects
    idx_lookup = {c["chunk_index"]: c for c in dense_chunks + sparse_chunks}

    return [idx_lookup[cid] for cid, _ in fused]
