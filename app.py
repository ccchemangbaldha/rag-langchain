import streamlit as st
import time
from uuid import uuid4

# Internal modules
from ui.upload import upload_files_widget
from services.preview import merge_files
from chunks.semantic_chunker import split_sentences, cluster_sentences
from embedding.preview_embedding import embed_sentences
from services.store import store_chunks
from services.hybrid import hybrid_rag
from services.retrieve_chunks import retrieve_chunks

# --- Page Config ---
st.set_page_config(page_title="RAG Chat Assistant", page_icon="🤖", layout="wide")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

if "current_upload_id" not in st.session_state:
    st.session_state.current_upload_id = None

# --- Sidebar: Configuration & Knowledge Base ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    st.info("Upload documents to start the chat.")
    
    # 1. File Upload Widget
    tmp_paths = upload_files_widget()

    # 2. Process Button Logic
    if tmp_paths and not st.session_state.processing_done:
        if st.button("🚀 Process Documents", type="primary"):
            with st.status("⚙️ Building Knowledge Base...", expanded=True) as status:
                
                st.write("🔗 Merging & Parsing Files...")
                resp = merge_files(tmp_paths)
                st.session_state["merged_name"] = resp["name"]
                st.session_state["merged_content"] = resp["content"]
                st.session_state["merged_files"] = resp["files_merged"]
                
                st.write("🧠 Semantic Chunking & Embedding...")
                sentences = split_sentences(resp["content"])
                vectors = embed_sentences(sentences)
                chunks = cluster_sentences(sentences, vectors, resp["files_merged"])
                
                st.write("💾 Storing Vectors (Pinecone)...")
                new_id = str(uuid4())
                store_chunks(new_id, resp["name"], chunks, vectors)
                
                # Save State
                st.session_state["current_upload_id"] = new_id
                st.session_state.processing_done = True
                
                status.update(label="✅ Ready to Chat!", state="complete", expanded=False)
                st.rerun()

    # 3. Status Display
    if st.session_state.processing_done:
        st.success(f"**Active ID:** `{st.session_state.current_upload_id}`")
        st.write(f"**Files:** {len(st.session_state.get('merged_files', []))}")
        st.write(f"**Total Chunks:** {len(st.session_state.get('chunks', []))}")
        
        if st.button("🔄 Reset / Clear", type="secondary"):
            st.session_state.clear()
            st.rerun()
            
    st.divider()
    st.caption("Settings")
    generate_viz = st.toggle("🎨 Generate AI Diagrams", value=False, help="Slower, uses DALL-E 3")

# --- Main Area: Chat Interface ---
st.title("💬 Intelligent Document Chat")
st.caption("🚀 Powered by RAG (Hybrid Search + Semantic Reranking)")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            st.image(message["image"], caption="Generated Illustration")

# 2. Handle User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Check if KB is ready
    if not st.session_state.processing_done:
        st.error("⚠️ Please upload and process documents in the sidebar first!")
        st.stop()

    # Append User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.status("🧠 Thinking...", expanded=False) as status:
            upload_id = st.session_state.current_upload_id
            
            st.write("🔍 Retrieving chunks...")
            # Retrieve Step
            results = retrieve_chunks(prompt, upload_id, limit=50, threshold=0.1)
            
            if not results:
                full_response = "I couldn't find any relevant information in the uploaded documents to answer that."
                image_url = None
                status.update(label="❌ No context found", state="error")
            else:
                st.write("✨ Synthesizing answer (GPT-4o)...")
                # Generation Step
                rag_response = hybrid_rag(
                    query=prompt, 
                    dense_chunks=results, 
                    final_top_k=8, 
                    enable_image=generate_viz
                )
                
                full_response = rag_response["answer"]
                confidence = rag_response["confidence"]
                image_url = rag_response.get("image_url")
                
                # Append Metadata footer
                if confidence < 0.4:
                    full_response += "\n\n> ⚠️ **Note:** *Confidence is low. Please verify with the source documents.*"
                
                status.update(label="✅ Response Generated", state="complete")

        # Stream/Display Result
        message_placeholder.markdown(full_response)
        if image_url:
            st.image(image_url, caption=f"Visual for: {prompt}")

    # Save Assistant Message to History
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "image": image_url
    })