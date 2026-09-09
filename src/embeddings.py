import logging
from typing import List

import numpy as np
import streamlit as st
from langchain_ollama import OllamaEmbeddings

from src.constants import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL
from src.utils import setup_logging

# Initialize logger
setup_logging()  # Configures logging for the application
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> OllamaEmbeddings:
    """
    Creates and caches a client for the host Ollama embedding model.

    Returns:
        OllamaEmbeddings: The Ollama embedding client.
    """
    logger.info("Using Ollama embedding model %s", OLLAMA_EMBEDDING_MODEL)
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
        num_gpu=1,
        num_ctx=1024,
        keep_alive=300,
    )


def generate_embeddings(chunks: List[str]) -> List[np.ndarray]:
    """
    Generates embeddings for a list of text chunks.

    Args:
        chunks (List[str]): List of text chunks.

    Returns:
        List[np.ndarray[Any, Any]]: List of embeddings as numpy arrays for each chunk.
    """
    embeddings = get_embedding_model().embed_documents(chunks)
    embeddings = [np.asarray(vector, dtype=np.float32) for vector in embeddings]
    logger.info(f"Generated embeddings for {len(chunks)} text chunks.")
    return embeddings
