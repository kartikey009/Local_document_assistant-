import logging
import os
import time

import streamlit as st
from PyPDF2 import PdfReader

from src.constants import TEXT_CHUNK_SIZE
from src.embeddings import generate_embeddings, get_embedding_model
from src.ingestion import (
    bulk_index_documents,
    create_index,
    delete_documents_by_document_name,
    list_document_names,
)
from src.utils import chunk_text, setup_logging

# Initialize logger
setup_logging()  # Set up centralized logging configuration
logger = logging.getLogger(__name__)

# Set page config with title, icon, and layout
st.set_page_config(page_title="Panda's Chatbot - Upload Documents", page_icon="📂")

# Custom CSS to style the page and sidebar
st.markdown(
    """
    <style>
    :root { --canvas: #201e1b; --ink: #faf9f5; --body: #d6d0c7; --muted: #a09d96; --hairline: #3a3631; --dark: #181715; --dark-soft: #302c27; --coral: #cc785c; --coral-dark: #a9583e; }
    .stApp { background: var(--canvas); color: var(--body); }
    .block-container { max-width: 1180px; padding: 2.5rem clamp(1rem, 4vw, 4rem) 5rem; }
    [data-testid="stSidebar"] { background: var(--dark); }
    [data-testid="stSidebar"] * { color: #faf9f5; }
    h1, h2, h3, h4 { color: var(--ink); font-family: Georgia, "Times New Roman", serif; font-weight: 400; letter-spacing: 0; }
    h1 { font-size: clamp(2.1rem, 5vw, 4rem); line-height: 1.05; }
    .kicker { color: var(--coral-dark); font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .intro { max-width: 42rem; color: var(--muted); font-size: 1.05rem; line-height: 1.6; margin-bottom: 2rem; }
    .upload-band { border: 1px dashed var(--coral); border-radius: 8px; background: #2a2723; padding: 1.25rem; margin: 1.5rem 0 2rem; }
    .upload-band strong { color: var(--ink); }
    .document-row { border-top: 1px solid var(--hairline); padding: 1rem 0; }
    .document-name { color: var(--ink); font-weight: 700; overflow-wrap: anywhere; }
    .document-meta { color: var(--muted); font-size: .86rem; margin-top: .25rem; }
    .stButton button { background: var(--coral); color: #fffdf9; border: 1px solid var(--coral-dark); border-radius: 8px; min-height: 44px; }
    .stButton button:hover { background: var(--coral-dark); color: #fffdf9; }
    @media (max-width: 767px) { .block-container { padding: 1.5rem 1rem 5rem; } h1 { font-size: 2.25rem; } .intro { font-size: 1rem; } .upload-band { padding: 1rem; } }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add a logo (replace with your own image file path or URL)
logo_path = "images/logo.png.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=220)
else:
    st.sidebar.markdown("### Logo Placeholder")
    logger.warning("Logo not found, displaying placeholder.")

# Sidebar header
st.sidebar.markdown(
    "<h2 style='text-align: center;'>Panda's Chatbot</h2>", unsafe_allow_html=True
)
st.sidebar.markdown(
    "<h4 style='text-align: center;'>Your Document Assistant</h4>",
    unsafe_allow_html=True,
)

# Footer
st.sidebar.markdown(
    """
    <div class="footer-text">
        © 2025 Panda's Chatbot
    </div>
    """,
    unsafe_allow_html=True,
)


def render_upload_page() -> None:
    """
    Renders the document upload page for users to upload and manage PDFs.
    Shows only the documents that are present in the OpenSearch index.
    """

    st.markdown('<div class="kicker">Local document desk</div>', unsafe_allow_html=True)
    st.title("Bring your documents in.")
    st.markdown(
        '<p class="intro">Add PDFs to the local workspace. Their passages are indexed in FAISS and stay on this machine.</p>',
        unsafe_allow_html=True,
    )
    # Placeholder for the loading spinner at the top
    model_loading_placeholder = st.empty()

    # Display the loading spinner at the top for loading the embedding model
    if "embedding_models_loaded" not in st.session_state:
        with model_loading_placeholder:
            with st.spinner("Loading models for document processing..."):
                get_embedding_model()
                st.session_state["embedding_models_loaded"] = True
        logger.info("Embedding models loaded.")
        model_loading_placeholder.empty()  # Clear the placeholder after loading

    UPLOAD_DIR = "uploaded_files"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    create_index()

    # Initialize or clear the documents list in session state
    st.session_state["documents"] = []

    document_names = list_document_names()
    logger.info("Retrieved document names from FAISS metadata.")

    # Load document information from the index
    for document_name in document_names:
        file_path = os.path.join(UPLOAD_DIR, document_name)
        if os.path.exists(file_path):
            reader = PdfReader(file_path)
            text = "".join([page.extract_text() for page in reader.pages])
            st.session_state["documents"].append(
                {"filename": document_name, "content": text, "file_path": file_path}
            )
        else:
            st.session_state["documents"].append(
                {"filename": document_name, "content": "", "file_path": None}
            )
            logger.warning(f"File '{document_name}' does not exist locally.")

    if "deleted_file" in st.session_state:
        st.success(
            f"The file '{st.session_state['deleted_file']}' was successfully deleted."
        )
        del st.session_state["deleted_file"]

    st.markdown(
        '<div class="upload-band"><strong>Drop PDFs here</strong><br><span>Upload one or several files to make them searchable.</span></div>',
        unsafe_allow_html=True,
    )

    # Allow users to upload PDF files
    uploaded_files = st.file_uploader(
        "Upload PDF documents", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Uploading and processing documents. Please wait..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name in document_names:
                    st.warning(
                        f"The file '{uploaded_file.name}' already exists in the index."
                    )
                    continue

                file_path = save_uploaded_file(uploaded_file)
                reader = PdfReader(file_path)
                text = "".join([page.extract_text() for page in reader.pages])
                chunks = chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=100)
                embeddings = generate_embeddings(chunks)

                documents_to_index = [
                    {
                        "doc_id": f"{uploaded_file.name}_{i}",
                        "text": chunk,
                        "embedding": embedding,
                        "document_name": uploaded_file.name,
                    }
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ]
                bulk_index_documents(documents_to_index)
                st.session_state["documents"].append(
                    {
                        "filename": uploaded_file.name,
                        "content": text,
                        "file_path": file_path,
                    }
                )
                document_names.append(uploaded_file.name)
                logger.info(f"File '{uploaded_file.name}' uploaded and indexed.")

        st.success("Files uploaded and indexed successfully!")

    if st.session_state["documents"]:
        st.markdown("### In this workspace")
        with st.expander("Manage Uploaded Documents", expanded=True):
            for idx, doc in enumerate(st.session_state["documents"], 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(
                        f'<div class="document-row"><div class="document-name">{idx:02d} &nbsp; {doc["filename"]}</div><div class="document-meta">{len(doc["content"]):,} characters extracted</div></div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    delete_button = st.button(
                        "Delete",
                        key=f"delete_{doc['filename']}_{idx}",
                        help=f"Delete {doc['filename']}",
                    )
                    if delete_button:
                        if doc["file_path"] and os.path.exists(doc["file_path"]):
                            try:
                                os.remove(doc["file_path"])
                                logger.info(
                                    f"Deleted file '{doc['filename']}' from filesystem."
                                )
                            except FileNotFoundError:
                                st.error(
                                    f"File '{doc['filename']}' not found in filesystem."
                                )
                                logger.error(
                                    f"File '{doc['filename']}' not found during deletion."
                                )
                        delete_documents_by_document_name(doc["filename"])
                        st.session_state["documents"].pop(idx - 1)
                        st.session_state["deleted_file"] = doc["filename"]
                        time.sleep(0.5)
                        st.rerun()
    else:
        st.markdown(
            '<div class="document-row"><div class="document-name">No documents yet</div><div class="document-meta">Your indexed PDFs will appear here after upload.</div></div>',
            unsafe_allow_html=True,
        )


def save_uploaded_file(uploaded_file) -> str:  # type: ignore
    """
    Saves an uploaded file to the local file system.

    Args:
        uploaded_file: The uploaded file to save.

    Returns:
        str: The file path where the uploaded file is saved.
    """
    UPLOAD_DIR = "uploaded_files"
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"File '{uploaded_file.name}' saved to '{file_path}'.")
    return file_path


if __name__ == "__main__":
    if "documents" not in st.session_state:
        st.session_state["documents"] = []
    render_upload_page()
