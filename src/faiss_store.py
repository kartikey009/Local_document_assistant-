import logging
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS

from src.constants import FAISS_INDEX_PATH
from src.embeddings import get_embedding_model
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_faiss_index() -> FAISS:
    """Load the local FAISS index and its metadata."""
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


def hybrid_search(
    query_text: str, query_embedding: List[float], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Run semantic search and preserve the result shape expected by the app."""
    del query_text
    try:
        store = load_faiss_index()
    except FileNotFoundError:
        logger.info("No FAISS index exists yet; returning no search results.")
        return []

    matches = store.similarity_search_with_score_by_vector(query_embedding, k=top_k)
    return [
        {
            "_score": 1.0 / (1.0 + float(distance)),
            "_source": {
                "text": document.page_content,
                "document_name": document.metadata.get("document_name", ""),
            },
        }
        for document, distance in matches
    ]