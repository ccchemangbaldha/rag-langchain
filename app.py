import streamlit as st
import time
from uuid import uuid4
import re

from ui.upload import upload_files_widget
from services.preview import merge_files
from chunks.semantic_chunker import create_smart_chunks
from embedding.preview_embedding import embed_sentences
from services.store import store_chunks
from services.hybrid import hybrid_rag
from services.retrieve_chunks import retrieve_chunks

st.set_page_config(
    page_title="Smart Study Buddy", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Chat Container */
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }
    /* User Message Bubble */
    div[data-testid="stChatMessage"][data-testid-user="true"] {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    /* Assistant Message Bubble */
    div[data-testid="stChatMessage"][data-testid-user="false"] {
        background-color: #f1f8e9;
        border-left: 5px solid #66bb6a;
    }
    /* Headers */
    h1, h2, h3 {
        font-family: 'Comic Sans MS', 'Chalkboard SE', sans-serif !important; 
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "👋 Hi there! I'm your AI Study Buddy. Upload your school notes, textbooks, or PDFs in the sidebar, and I'll help you learn!",
        "image": None
    })

if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

if "current_upload_id" not in st.session_state:
    st.session_state.current_upload_id = None

with st.sidebar:
    st.header("🎒 My Backpack")
    st.markdown("Add your study materials here:")
    
    tmp_paths = upload_files_widget()

    if tmp_paths and not st.session_state.processing_done:
        st.divider()
        if st.button("🚀 Start Studying!", type="primary", use_container_width=True):
            with st.status("⚙️ Organizing your notes...", expanded=True) as status:
                
                st.write("📖 Reading files...")
                resp = merge_files(tmp_paths)
                
                st.write("✂️ Creating smart study chunks...")
                chunks = create_smart_chunks(resp["content"], resp["files_merged"])
                
                st.write("🧠 Memorizing content...")
                text_list = [c["text"] for c in chunks]
                vectors = embed_sentences(text_list)
                
                st.write("💾 Saving to Brain (Database)...")
                new_id = str(uuid4())
                store_chunks(new_id, resp["name"], chunks, vectors)
                
                st.session_state["current_upload_id"] = new_id
                st.session_state.processing_done = True
                st.session_state["total_chunks"] = len(chunks)
                
                status.update(label="✅ Ready to learn!", state="complete", expanded=False)
                time.sleep(1)
                st.rerun()

    if st.session_state.processing_done:
        st.success(f"📚 **Study Set Active**")
        st.caption(f"ID: `{st.session_state.current_upload_id}`")
        st.markdown(f"**Knowledge chunks:** {st.session_state.get('total_chunks', 0)}")
        
        if st.button("🗑️ Clear & Start Over", type="secondary", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    st.divider()
    st.markdown("### 🎨 Creativity Mode")
    generate_viz = st.toggle("✨ Draw diagrams for me", value=True, help="I will draw a picture if I find a good answer!")

st.title("🎓 Smart Study Buddy")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image"):
            st.image(message["image"], caption="🎨 Here is a visual helper!", use_container_width=True)

if prompt := st.chat_input("Ex: What is Cybersecurity?"):
    
    if not st.session_state.processing_done:
        st.warning("⚠️ Wait! Your backpack is empty. Please upload some files in the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🤔 Thinking hard..."):
            upload_id = st.session_state.current_upload_id
            
            # --- MODIFIED: Multi-Question Retrieval Logic ---
            # Split the prompt into individual sentences/questions
            sub_queries = re.split(r'(?<=[.?!])\s+', prompt)
            sub_queries = [q.strip() for q in sub_queries if len(q.strip()) > 5]

            all_combined_results = []
            seen_chunk_ids = set()

            # If re.split didn't find multiple sentences, use the full prompt
            if not sub_queries:
                sub_queries = [prompt]

            for query_part in sub_queries:
                # Retrieve chunks for each specific part of the user's input
                part_results = retrieve_chunks(query_part, upload_id, limit=15, threshold=0.1)
                
                for res in part_results:
                    # Deduplicate based on chunk_index
                    if res["chunk_index"] not in seen_chunk_ids:
                        all_combined_results.append(res)
                        seen_chunk_ids.add(res["chunk_index"])
            # ------------------------------------------------
            
            if not all_combined_results:
                full_response = "I looked through your notes, but I couldn't find anything about those topics. 🤷‍♂️"
                image_url = None
            else:
                # Use the combined chunks for the final RAG generation
                rag_response = hybrid_rag(
                    query=prompt, 
                    dense_chunks=all_combined_results, 
                    final_top_k=10, # Increased k to handle multiple topics
                    enable_image=generate_viz
                )
                
                full_response = rag_response["answer"]
                image_url = rag_response.get("image_url")
                
                if rag_response["confidence"] < 0.35 and "I don't know" not in full_response:
                    full_response += "\n\n> 🧐 *I'm not 100% sure, so please double-check your textbooks!*"

        message_placeholder.markdown(full_response)
        if image_url:
            st.image(image_url, caption=f"🎨 Visual: {prompt}")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "image": image_url
    })