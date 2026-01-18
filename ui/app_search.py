import streamlit as st
from services.retrieve_chunks import retrieve_chunks

st.set_page_config(page_title="RAG Semantic Search", layout="wide")
st.title("🔎 Semantic Chunk Search")

query = st.text_input("Enter query:")
upload_id = st.text_input("Upload ID (filter):")

limit = st.number_input("Limit", 1, 50, 5)
threshold = st.slider("Threshold", 0.0, 1.0, 0.0, 0.01)

if st.button("Search") and query.strip() and upload_id.strip():
    results = retrieve_chunks(query, upload_id=upload_id, limit=limit, threshold=threshold)

    st.write(f"{len(results)} chunks found")

    for r in results:
        with st.expander(f"Score {r['score']} | Chunk {r['chunk_index']}"):
            st.text(r["text"])
            st.json({
                "source_files": r["source_files"],
                "tokens": r["tokens"]
            })
