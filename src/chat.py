import logging
from typing import Dict, Iterable, List, Optional

import ollama
import streamlit as st

from src.constants import ASSYMETRIC_EMBEDDING, OLLAMA_BASE_URL, OLLAMA_MODEL_NAME
from src.embeddings import get_embedding_model
from src.faiss_store import hybrid_search
from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


def is_casual_message(query: str) -> bool:
    """Return True for conversational messages that do not need document retrieval."""
    normalized = query.strip().lower().rstrip("!?.")
    return normalized in {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
    }


@st.cache_resource(show_spinner=False)
def ensure_model_pulled(model: str) -> bool:
    """
    Ensures that the specified model is pulled and available locally.

    Args:
        model (str): The name of the model to ensure is available.

    Returns:
        bool: True if the model is available or successfully pulled, False if an error occurs.
    """
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        available_models = client.list()
        model_names = {
            getattr(item, "model", getattr(item, "name", "")).split(":")[0]
            for item in getattr(available_models, "models", [])
        }
        if model.split(":")[0] not in model_names:
            logger.error(
                "Model %s is not installed. Run 'ollama pull %s' before starting the app.",
                model,
                model,
            )
            return False
        logger.info("Model %s is available locally.", model)
    except ollama.ResponseError as e:
        logger.error(f"Error checking or pulling model: {e.error}")
        return False
    return True


def run_llama_streaming(prompt: str, temperature: float) -> Optional[Iterable[str]]: 
    """
    Uses Ollama's Python library to run the LLaMA model with streaming enabled.

    Args:
        prompt (str): The prompt to send to the model.
        temperature (float): The response generation temperature.

    Returns:
        Optional[Iterable[str]]: A generator yielding response chunks as strings, or None if an error occurs.
    """

    try:
        # Now attempt to stream the response from the model
        logger.info("Streaming response from LLaMA model.")
        client = ollama.Client(host=OLLAMA_BASE_URL)
        stream = client.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={
                "temperature": min(temperature, 0.2),
                "num_gpu": 0,
                "num_ctx": 1024,
            },
            keep_alive=300,
        )
    except ollama.ResponseError as e:
        logger.error(f"Error during streaming: {e.error}")
        return None

    return stream


def prompt_template(query: str, context: str, history: List[Dict[str, str]]) -> str:
    """
    Builds the prompt with context, conversation history, and user query.

    Args:
        query (str): The user's query.
        context (str): Context text gathered from hybrid search.
        history (List[Dict[str, str]]): Conversation history to include in the prompt.

    Returns:
        str: Constructed prompt for Ollama model.
    """
    prompt = (
        "You answer only from the provided document context. "
        "Do not use outside knowledge. Do not invent instructions, steps, or facts. "
        "If the context does not answer the question, reply exactly: "
        "I could not find that information in the uploaded documents.\n\n"
    )
    prompt += "Document context:\n"
    prompt += context[:6000] if context else "(No relevant document context was found.)"
    prompt += "\n\n"

    if history:
        prompt += "Conversation History:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        prompt += "\n"

    prompt += f"User: {query}\nAssistant:"
    logger.info("Prompt constructed with context and conversation history.")
    return prompt


def generate_response_streaming(
    query: str,
    use_hybrid_search: bool,
    num_results: int,
    temperature: float,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Iterable[str]]:
    """
    Generates a chatbot response by performing hybrid search and incorporating conversation history.

    Args:
        query (str): The user's query.
        use_hybrid_search (bool): Whether to use hybrid search for context.
        num_results (int): The number of search results to include in the context.
        temperature (float): The temperature for the response generation.
        chat_history (Optional[List[Dict[str, str]]]): List of chat history messages.

    Returns:
        Optional[Iterable[str]]: A generator yielding response chunks as strings, or None if an error occurs.
    """
    chat_history = chat_history or []
    max_history_messages = 10
    # The page already appends the current user message before calling this function.
    history = chat_history[:-1][-max_history_messages:]
    context = ""
    st.session_state["last_sources"] = []

    if is_casual_message(query):
        return iter(
            [
                "Hello! Ask me a question about your uploaded documents, "
                "and I will search them for an answer."
            ]
        )

    # Include hybrid search results if enabled
    if use_hybrid_search:
        logger.info("Performing hybrid search.")
        if ASSYMETRIC_EMBEDDING:
            prefixed_query = f"passage: {query}"
        else:
            prefixed_query = f"{query}"
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.embed_query(prefixed_query)
        search_results = hybrid_search(query, query_embedding, top_k=num_results)
        logger.info("Hybrid search completed.")

        # Collect text from search results
        for i, result in enumerate(search_results, 1):
            document_name = result["_source"].get("document_name", "uploaded document")
            st.session_state["last_sources"].append(
                {
                    "document_name": document_name,
                    "score": result.get("_score", 0.0),
                    "text": result["_source"]["text"],
                }
            )
            context += (
                f"Document {i} ({document_name}):\n"
                f"{result['_source']['text']}\n\n"
            )

    # Generate prompt using the prompt_template function
    prompt = prompt_template(query, context, history)

    return run_llama_streaming(prompt, temperature)
