# chunks/semantic_chunker.py
import spacy
import hdbscan

nlp = spacy.load("en_core_web_sm")

MAX_TOKENS = 400
OVERLAP_TOKENS = 40

def count_tokens(text: str):
    return len(text.split())

def split_sentences(content: str):
    doc = nlp(content)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

def cluster_sentences(sentences, vectors, source_files):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    labels = clusterer.fit_predict(vectors)

    raw = {}
    for idx, (sent, lbl) in enumerate(zip(sentences, labels)):
        if lbl == -1:
            lbl = f"noise-{idx}"
        raw.setdefault(lbl, []).append(sent)

    chunks = []
    chunk_index = 0

    for lbl, sent_list in raw.items():
        current = []
        current_tokens = 0

        for s in sent_list:
            t = count_tokens(s)

            if current and current_tokens + t > MAX_TOKENS:
                text = " ".join(current)
                chunks.append({
                    "cluster_id": lbl,
                    "chunk_index": chunk_index,
                    "text": text,
                    "tokens": current_tokens,
                    "source_files": source_files,
                })
                chunk_index += 1

                overlap = text.split()[-OVERLAP_TOKENS:]
                current = [" ".join(overlap)]
                current_tokens = len(overlap)

            current.append(s)
            current_tokens += t

        if current:
            text = " ".join(current)
            chunks.append({
                "cluster_id": lbl,
                "chunk_index": chunk_index,
                "text": text,
                "tokens": current_tokens,
                "source_files": source_files,
            })
            chunk_index += 1

    return chunks
