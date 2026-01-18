# app.py
import streamlit as st
from uuid import uuid4

# Import your existing modules
from ui.upload import upload_files_widget
from services.preview import merge_files
from chunks.semantic_chunker import split_sentences, cluster_sentences
from embedding.preview_embedding import embed_sentences
from services.store import store_chunks
from services.hybrid import hybrid_rag

# NEW IMPORT: Retrieval service
from services.retrieve_chunks import retrieve_chunks

st.set_page_config(page_title="RAG Preview Pipeline", layout="wide")
st.title("📚 RAG Preview — Auto-Pipeline")

# ---------------------------------------
# 1. Upload & Auto-Process Phase
# ---------------------------------------
st.header("1. Upload Documents")
tmp_paths = upload_files_widget()

# Reset processing state if files are cleared/removed
if not tmp_paths:
    st.session_state["processing_done"] = False

# Initialize state if not present
if "processing_done" not in st.session_state:
    st.session_state["processing_done"] = False

# AUTOMATION: If files exist and we haven't processed them yet, run the full pipeline
if tmp_paths and not st.session_state["processing_done"]:
    
    # Use st.status to show the user the progress of the automated steps
    with st.status("🚀 Auto-Processing: Merging → Chunking → Storing...", expanded=True) as status:
        
        # --- Step A: Merge ---
        st.write("🔗 Merging Files...")
        resp = merge_files(tmp_paths)
        st.session_state["merged_name"] = resp["name"]
        st.session_state["merged_content"] = resp["content"]
        st.session_state["merged_files"] = resp["files_merged"]
        
        # --- Step B: Chunk & Embed ---
        st.write("🧩 Chunking & Embedding...")
        content = st.session_state["merged_content"]
        files = st.session_state["merged_files"]
        
        sentences = split_sentences(content)
        vectors = embed_sentences(sentences)
        chunks = cluster_sentences(sentences, vectors, files)
        
        st.session_state["chunks"] = chunks
        st.session_state["vectors"] = vectors
        
        # --- Step C: Store ---
        st.write("📦 Storing in Qdrant...")
        new_id = str(uuid4())
        upload_name = st.session_state["merged_name"]
        
        store_chunks(new_id, upload_name, chunks, vectors)
        
        # Save the ID and mark processing as complete
        st.session_state["current_upload_id"] = new_id
        st.session_state["processing_done"] = True
        
        status.update(label="✅ Processing Complete! Data ready for search.", state="complete", expanded=False)

# Display Summary after processing
if st.session_state.get("processing_done"):
    st.success(f"Files Processed Successfully! Active Upload ID: `{st.session_state['current_upload_id']}`")
    
    with st.expander("📝 View Processed Content Details"):
        st.write(f"**Merged Name:** {st.session_state.get('merged_name')}")
        st.write(f"**Total Chunks:** {len(st.session_state.get('chunks', []))}")
        st.text_area("Content Preview", st.session_state.get("merged_content", "")[:1000], height=150)
        
        if "chunks" in st.session_state:
            st.divider()
            st.write("**Sample Chunks:**")
            for i, c in enumerate(st.session_state["chunks"][:3]):
                st.markdown(f"**Chunk {c['chunk_index']}** ({c['tokens']} tokens)")
                st.text(c['text'])
                st.divider()

# ---------------------------------------
# 2. Search / Retrieval Phase
# ---------------------------------------
st.header("2. Test Retrieval")
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

if st.button("🔍 Call LLM Search"):
    if not target_id or not user_query:
        st.error("Please provide both an Upload ID and a Query.")
    else:
        with st.spinner("Searching..."):
            results = retrieve_chunks(user_query, target_id, limit=12, threshold=0.1)

        if not results:
            st.warning("No matches found.")
        else:
            rag = hybrid_rag(query=user_query, dense_chunks=results, final_top_k= 5 )

            st.subheader("🧠 RAG Answer")
            st.write(rag["answer"])

            st.subheader("📊 Confidence")
            st.write(rag["confidence"])

            st.subheader("📎 Citations")
            st.write(rag["citations"])

            with st.expander("🔍 Retrieved Chunks"):
                for r in results:
                    st.text(f"[{r['chunk_index']}] {r['text']}")