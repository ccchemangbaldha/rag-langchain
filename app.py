# app.py
import streamlit as st

from ui.upload import upload_files_widget
from services.preview import merge_files
from chunks.semantic_chunker import split_sentences, cluster_sentences
from embedding.preview_embedding import embed_sentences
from services.store import store_chunks

st.set_page_config(page_title="RAG Preview Pipeline", layout="wide")
st.title("📚 RAG Preview — Merge → Chunk → Embed → Store")

# ---------------------------------------
# 1. Upload Phase
# ---------------------------------------
tmp_paths = upload_files_widget()

# ---------------------------------------
# 2. Merge Phase
# ---------------------------------------
if tmp_paths and st.button("🔗 Merge Files"):
    resp = merge_files(tmp_paths)

    st.session_state["merged_name"] = resp["name"]
    st.session_state["merged_content"] = resp["content"]
    st.session_state["merged_files"] = resp["files_merged"]

    st.success(f"Merged into: {resp['name']}")
    with st.expander("📂 Files Merged"):
        st.json(resp["files_merged"])

    with st.expander("📝 Merged Content"):
        st.text_area("Content", resp["content"], height=300)


# ---------------------------------------
# 3. Chunk + Embedding Phase
# ---------------------------------------
if (
    "merged_content" in st.session_state
    and st.button("🧩 Preview Chunks + Embeddings")
):
    content = st.session_state["merged_content"]
    files = st.session_state["merged_files"]

    sentences = split_sentences(content)
    vectors = embed_sentences(sentences)
    chunks = cluster_sentences(sentences, vectors, files)

    st.session_state["chunks"] = chunks
    st.session_state["vectors"] = vectors

    st.success(f"Semantic Chunking Completed — {len(chunks)} Chunks")

    with st.expander("🧩 Chunk Preview"):
        for c in chunks:
            st.text(f"[{c['chunk_index']}] {c['tokens']} tokens\n{c['text']}\n---")

    with st.expander("🧬 Embedding Sample"):
        st.write(f"Vectors: {len(vectors)} total")
        st.json(vectors[0][:20])


# ---------------------------------------
# 4. Store in Qdrant (Hybrid Retrieval Ready)
# ---------------------------------------
if (
    "chunks" in st.session_state
    and "vectors" in st.session_state
    and st.button("📦 Store in Qdrant")
):
    from uuid import uuid4
    upload_id = str(uuid4())
    upload_name = st.session_state["merged_name"]

    store_chunks(upload_id, upload_name, st.session_state["chunks"], st.session_state["vectors"])

    st.success(f"Stored in Qdrant! upload_id = {upload_id}")
