# ui/upload.py
import streamlit as st
import tempfile

def upload_files_widget():
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "pptx", "ppt", "docx", "doc", "txt", "zip","md"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        return None

    st.write(f"{len(uploaded_files)} file(s) selected")

    tmp_paths = []
    for f in uploaded_files:
        suffix = "." + f.name.split(".")[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(f.getbuffer())
        tmp_paths.append(tmp.name)

    return tmp_paths
