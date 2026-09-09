import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import FAISS

from src.constants import FAISS_INDEX_PATH
from src.embeddings import get_embedding_model
from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


def _index_exists() -> bool:
    path = Path(FAISS_INDEX_PATH)
    return (path / "index.faiss").exists() and (path / "index.pkl").exists()


def create_index() -> None:
    """Create the local FAISS directory when the app starts."""
    Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)


def delete_index() -> None:
    """Delete the persisted FAISS index."""
    path = Path(FAISS_INDEX_PATH)
    if path.exists():
        shutil.rmtree(path)
        logger.info("Deleted FAISS index at %s", FAISS_INDEX_PATH)


def bulk_index_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    """
    Adds document chunks to the persisted FAISS index.

    Args:
        documents (List[Dict[str, Any]]): List of document dictionaries with 'doc_id', 'text', 'embedding', and 'document_name'.

    Returns:
        Tuple[int, List[Any]]: Tuple with the number of successfully indexed documents and a list of any errors.
    """
    if not documents:
        return 0, []

    text_embeddings = [
        (doc["text"], doc["embedding"].tolist()) for doc in documents
    ]
    metadatas = [{"document_name": doc["document_name"]} for doc in documents]
    ids = [doc["doc_id"] for doc in documents]
    embedding_model = get_embedding_model()

    if _index_exists():
        store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        store.add_embeddings(text_embeddings, metadatas=metadatas, ids=ids)
    else:
        store = FAISS.from_embeddings(
            text_embeddings,
            embedding=embedding_model,
            metadatas=metadatas,
            ids=ids,
        )

    Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
    store.save_local(FAISS_INDEX_PATH)
    logger.info("Persisted %d chunks to FAISS.", len(documents))
    return len(documents), []


def delete_documents_by_document_name(document_name: str) -> Dict[str, Any]:
    """
    Deletes FAISS entries whose metadata matches the document name.

    Args:
        document_name (str): Name of the document to delete.

    Returns:
        Dict[str, Any]: Number of deleted chunks.
    """
    if not _index_exists():
        return {"deleted": 0}

    store = FAISS.load_local(
        FAISS_INDEX_PATH,
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )
    ids = [
        doc_id
        for doc_id, document in store.docstore._dict.items()
        if document.metadata.get("document_name") == document_name
    ]
    if ids:
        store.delete(ids)
        store.save_local(FAISS_INDEX_PATH)
    logger.info("Deleted %d chunks for '%s'.", len(ids), document_name)
    return {"deleted": len(ids)}


def list_document_names() -> List[str]:
    """Return unique document names stored in FAISS metadata."""
    if not _index_exists():
        return []
    store = FAISS.load_local(
        FAISS_INDEX_PATH,
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )
    return sorted(
        {
            document.metadata.get("document_name", "")
            for document in store.docstore._dict.values()
            if document.metadata.get("document_name")
        }
    )
