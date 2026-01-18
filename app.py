# app.py
import streamlit as st
from uuid import uuid4

# Import your existing modules
from ui.upload import upload_files_widget
from services.preview import merge_files
from chunks.semantic_chunker import split_sentences, cluster_sentences
from embedding.preview_embedding import embed_sentences
from services.store import store_chunks

# NEW IMPORT: Retrieval service
from services.retrieve_chunks import retrieve_chunks

st.set_page_config(page_title="RAG Preview Pipeline", layout="wide")
st.title("📚 RAG Preview — Merge → Chunk → Embed → Store → Search")

# ---------------------------------------
# 1. Upload Phase
# ---------------------------------------
st.header("1. Upload Documents")
tmp_paths = upload_files_widget()

# ---------------------------------------
# 2. Merge Phase
# ---------------------------------------
st.header("2. Merge")
if tmp_paths and st.button("🔗 Merge Files"):
    resp = merge_files(tmp_paths)
    st.session_state["merged_name"] = resp["name"]
    st.session_state["merged_content"] = resp["content"]
    st.session_state["merged_files"] = resp["files_merged"]
    st.success(f"Merged into: {resp['name']}")

if "merged_content" in st.session_state:
    with st.expander("📝 View Merged Content"):
        st.text_area("Content", st.session_state["merged_content"], height=150)

# ---------------------------------------
# 3. Chunk + Embedding Phase
# ---------------------------------------
st.header("3. Chunk & Embed")
if "merged_content" in st.session_state:
    if st.button("🧩 Preview Chunks + Embeddings"):
        content = st.session_state["merged_content"]
        files = st.session_state["merged_files"]

        with st.spinner("Chunking and Embedding..."):
            sentences = split_sentences(content)
            vectors = embed_sentences(sentences)
            chunks = cluster_sentences(sentences, vectors, files)

        st.session_state["chunks"] = chunks
        st.session_state["vectors"] = vectors

        st.success(f"Processing Complete — {len(chunks)} Chunks generated.")

    if "chunks" in st.session_state:
        with st.expander("🧩 View Sample Chunks"):
            for i, c in enumerate(st.session_state["chunks"][:3]):
                st.markdown(f"**Chunk {c['chunk_index']}** ({c['tokens']} tokens)")
                st.text(c['text'])
                st.divider()

# ---------------------------------------
# 4. Store in Qdrant
# ---------------------------------------
st.header("4. Store Vectors")
if "chunks" in st.session_state and "vectors" in st.session_state:
    if st.button("📦 Store in Qdrant"):
        new_id = str(uuid4())
        upload_name = st.session_state["merged_name"]
        
        with st.spinner("Upserting to Qdrant..."):
            store_chunks(new_id, upload_name, st.session_state["chunks"], st.session_state["vectors"])
        
        # Save the ID to session state so we can use it in search immediately
        st.session_state["current_upload_id"] = new_id
        st.success(f"Stored successfully! Upload ID: `{new_id}`")

# ---------------------------------------
# 5. Search / Retrieval Phase (NEW)
# ---------------------------------------
st.header("5. Test Retrieval")
st.info("Search for nearest chunks within a specific Upload ID.")

col1, col2 = st.columns([1, 3])

# Auto-fill the ID if we just stored data
default_id = st.session_state.get("current_upload_id", "")

with col1:
    target_id = st.text_input("Target Upload ID", value=default_id)
with col2:
    user_query = st.text_input("Enter Search Query")

if st.button("🔍 Search"):
    if not target_id or not user_query:
        st.error("Please provide both an Upload ID and a Query.")
    else:
        with st.spinner("Searching..."):
            results = retrieve_chunks(
                query=user_query, 
                upload_id=target_id, 
                limit=8,
                threshold=0.5
            )
        
        if results:
            st.write(f"Found {len(results)} matches:")
            for r in results:
                with st.container(border=True):
                    st.markdown(f"**Score:** `{r['score']}` | **Chunk:** `{r['chunk_index']}`")
                    st.markdown(f"_{r['text']}_")
                    st.caption(f"Source: {r['source_files']}")
        else:
            st.warning("No matches found. Check your Upload ID.")