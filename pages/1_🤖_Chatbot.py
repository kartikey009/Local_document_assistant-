import logging
import os

import streamlit as st

from src.chat import (  # type: ignore
    ensure_model_pulled,
    generate_response_streaming,
)
from src.constants import OLLAMA_MODEL_NAME
from src.embeddings import get_embedding_model
from src.ingestion import create_index
from src.utils import setup_logging

# Initialize logger
setup_logging()  # Configures logging for the application
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(page_title="Panda's Chatbot", page_icon="🤖")

# Apply custom CSS
st.markdown(
    """
    <style>
    :root { --canvas: #201e1b; --ink: #faf9f5; --body: #d6d0c7; --muted: #a09d96; --hairline: #3a3631; --card: #2a2723; --dark: #181715; --dark-soft: #302c27; --coral: #cc785c; --coral-dark: #a9583e; --teal: #5db8a6; }
    .stApp { background: var(--canvas); color: var(--body); }
    [data-testid="stHeader"] { background: var(--canvas); }
    .block-container { max-width: 1040px; padding: 3rem clamp(1rem, 4vw, 3.5rem) 7rem; }
    [data-testid="stBottom"] { background: var(--dark); border-top: 1px solid var(--dark-soft); padding: .85rem 1rem 1rem; }
    [data-testid="stBottom"] [data-testid="stChatInput"] { max-width: 760px; margin: 0 auto; }
    [data-testid="stSidebar"] { background: var(--dark); border-right: 1px solid var(--dark-soft); }
    [data-testid="stSidebar"] * { color: #faf9f5; }
    [data-testid="stSidebar"] [data-testid="stSlider"] label,
    [data-testid="stSidebar"] [data-testid="stSlider"] span { color: #faf9f5 !important; }
    [data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div { background: #3a3631 !important; }
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] { background: var(--coral) !important; border-color: #faf9f5 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { color: #faf9f5; font-family: Georgia, "Times New Roman", serif; font-size: 1.35rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 { color: var(--muted); font-family: inherit; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }
    [data-testid="stSidebar"] [data-testid="stImage"] { margin: 0 auto 1rem; }
    h1, h2, h3 { color: var(--ink); font-family: Georgia, "Times New Roman", serif; font-weight: 400; letter-spacing: 0; }
    h1 { font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.08; margin-bottom: .5rem; }
    h2 { font-size: 1.75rem; }
    p, label, .stMarkdown { color: var(--body); }
    .status-strip { display: flex; gap: .6rem; flex-wrap: wrap; margin: 0 0 1.5rem; }
    .status-chip { border: 1px solid var(--hairline); border-radius: 999px; padding: .35rem .7rem; color: var(--muted); font-size: .76rem; background: var(--card); }
    .status-chip.ready { border-color: #3f756a; color: #9ed8cb; background: #263b36; }
    .empty-state { border-top: 1px solid var(--hairline); border-bottom: 1px solid var(--hairline); padding: 1.75rem 0 2rem; margin: 0 0 1.5rem; }
    .empty-state h2 { margin: 0 0 .5rem; }
    .empty-state p { max-width: 42rem; margin: 0; color: var(--muted); line-height: 1.6; }
    [data-testid="stChatMessage"] { border: 1px solid var(--hairline); border-radius: 8px; background: var(--card); padding: 1rem; margin: .75rem 0; max-width: 54rem; }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color: var(--body); }
    [data-testid="stChatInput"] { border: 1px solid #65564c; border-radius: 8px; background: #2a2723; box-shadow: 0 12px 28px rgba(0,0,0,.22); }
    [data-testid="stChatInput"] textarea { color: #faf9f5 !important; caret-color: var(--coral); }
    [data-testid="stChatInput"] textarea::placeholder { color: #b9b0a6 !important; opacity: 1; }
    [data-testid="stChatInput"] textarea:focus { box-shadow: 0 0 0 2px rgba(204,120,92,.35); }
    [data-testid="stChatInput"] button { color: var(--coral) !important; }
    .source-heading { border-top: 1px solid var(--hairline); margin-top: 1.5rem; padding-top: 1rem; color: var(--muted); font-size: .76rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .source-card { background: var(--dark); color: #faf9f5; border-radius: 8px; padding: 1rem; margin: .6rem 0; max-width: 54rem; }
    .source-name { color: #faf9f5; font-weight: 700; overflow-wrap: anywhere; }
    .source-score { color: #d6a18e; font-size: .76rem; }
    .source-text { color: #c9c5bc; font-size: .86rem; line-height: 1.5; margin-top: .45rem; }
    .typing-bubble { display: inline-flex; align-items: center; gap: .28rem; background: var(--dark); border-radius: 999px; padding: .7rem .9rem; }
    .typing-dot { width: .42rem; height: .42rem; border-radius: 50%; background: var(--coral); animation: typing-pulse 1.1s ease-in-out infinite; }
    .typing-dot:nth-child(2) { animation-delay: .14s; }
    .typing-dot:nth-child(3) { animation-delay: .28s; }
    @keyframes typing-pulse { 0%, 60%, 100% { opacity: .35; transform: translateY(0); } 30% { opacity: 1; transform: translateY(-.18rem); } }
    .stButton button { background: var(--coral); color: #fffdf9; border: 1px solid var(--coral-dark); border-radius: 8px; min-height: 44px; }
    .stButton button:hover { background: var(--coral-dark); color: #fffdf9; }
    [data-testid="stChatInput"] textarea { min-height: 44px; font-size: 1rem; }
    @media (min-width: 1024px) { [data-testid="stSidebar"] { min-width: 18rem; } .source-card:hover { background: var(--dark-soft); } }
    @media (min-width: 768px) and (max-width: 1023px) { .block-container { padding: 2rem 2rem 4rem; } .intro { max-width: 36rem; } [data-testid="stChatMessage"], .source-card { max-width: 48rem; } }
    @media (max-width: 767px) { .block-container { padding: 1.5rem 1rem 6rem; } h1 { font-size: 2.25rem; } .status-strip { gap: .4rem; margin-bottom: 1rem; } .status-chip { padding: .45rem .6rem; } [data-testid="stChatMessage"] { padding: .75rem; margin: .6rem 0; } .source-card { padding: .85rem; } [data-testid="stBottom"] { padding: .65rem .75rem .75rem; } [data-testid="stChatInput"] { bottom: .5rem; } [data-testid="stChatInput"] textarea { font-size: 16px; } [data-testid="stSidebar"] .stSlider, [data-testid="stSidebar"] .stNumberInput { width: 100%; } }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)
logger.info("Custom CSS applied.")


# Main chatbot page rendering function
def render_chatbot_page() -> None:
    model_loading_placeholder = st.empty()

    # Initialize session state variables for chatbot settings
    if "use_hybrid_search" not in st.session_state:
        st.session_state["use_hybrid_search"] = True
    if "num_results" not in st.session_state:
        st.session_state["num_results"] = 5
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.7

    create_index()

    if "last_sources" not in st.session_state:
        st.session_state["last_sources"] = []

    st.markdown(
        '<div class="empty-state"><h1>Start with a document question</h1><p>Ask about a role, requirement, date, or passage in the PDFs you uploaded. Retrieved sources will appear below each answer.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="status-strip"><span class="status-chip ready">FAISS index ready</span><span class="status-chip">Ollama on this computer</span><span class="status-chip">Source-first answers</span></div>',
        unsafe_allow_html=True,
    )

    # Sidebar settings for hybrid search toggle, result count, and temperature
    st.session_state["use_hybrid_search"] = st.sidebar.checkbox(
        "Enable RAG mode", value=st.session_state["use_hybrid_search"]
    )
    st.session_state["num_results"] = st.sidebar.number_input(
        "Number of Results in Context Window",
        min_value=1,
        max_value=10,
        value=st.session_state["num_results"],
        step=1,
    )
    st.session_state["temperature"] = st.sidebar.slider(
        "Response Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.1,
    )

    # Display logo or placeholder
    logo_path = "images/logo.png.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=220)
        logger.info("Logo displayed.")
    else:
        st.sidebar.markdown("### Logo Placeholder")
        logger.warning("Logo not found, displaying placeholder.")

    # Sidebar headers and footer
    st.sidebar.markdown(
        "<h2 style='text-align: center;'>Panda's Chatbot</h2>", unsafe_allow_html=True
    )
    st.sidebar.markdown(
        "<h4 style='text-align: center;'>Your Document Assistant</h4>",
        unsafe_allow_html=True,
    )

    # Footer text
    st.sidebar.markdown(
        """
        <div class="footer-text">
            © 2025 Panda's Chatbot
        </div>
        """,
        unsafe_allow_html=True,
    )
    logger.info("Sidebar configured with headers and footer.")

    # Load models if not already loaded
    if "embedding_models_loaded" not in st.session_state:
        with model_loading_placeholder:
            with st.spinner("Checking local models..."):
                get_embedding_model()
                model_ready = ensure_model_pulled(OLLAMA_MODEL_NAME)
                st.session_state["embedding_models_loaded"] = model_ready
                if not model_ready:
                    st.error(
                        f"Ollama model '{OLLAMA_MODEL_NAME}' is not installed. "
                        f"Run `ollama pull {OLLAMA_MODEL_NAME}` and reload the page."
                    )
        logger.info("Embedding model ready.")
        model_loading_placeholder.empty()

    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if not st.session_state["chat_history"]:
        st.caption("Your conversation will appear here.")

    # Display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process user input and generate response
    if prompt := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        logger.info("User input received.")

        # Generate response from assistant
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response_placeholder = st.empty()
                response_text = ""
                response_placeholder.markdown(
                    '<div class="typing-bubble" aria-label="Assistant is thinking"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>',
                    unsafe_allow_html=True,
                )

                response_stream = generate_response_streaming(
                    prompt,
                    use_hybrid_search=st.session_state["use_hybrid_search"],
                    num_results=st.session_state["num_results"],
                    temperature=st.session_state["temperature"],
                    chat_history=st.session_state["chat_history"],
                )

            # Stream response content if response_stream is valid
            if response_stream is not None:
                try:
                    for chunk in response_stream:
                        if isinstance(chunk, str):
                            piece = chunk
                        elif isinstance(chunk, dict):
                            piece = chunk.get("message", {}).get("content", "")
                        else:
                            message = getattr(chunk, "message", None)
                            piece = getattr(message, "content", "")

                        if piece:
                            response_text += piece
                            response_placeholder.markdown(response_text + "▌")
                except Exception as error:
                    logger.exception("Ollama response stream failed.")
                    st.error(f"Ollama could not generate a response: {error}")

            response_placeholder.markdown(response_text)
            if response_text:
                st.success("Answer generated from your local workspace.")
            else:
                st.error("No answer was returned. Check that Ollama is running and try again.")
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response_text}
            )
            logger.info("Response generated and displayed.")

            sources = st.session_state.get("last_sources", [])
            if sources:
                st.markdown('<div class="source-heading">Retrieved sources</div>', unsafe_allow_html=True)
                for source in sources:
                    st.markdown(
                        f'<div class="source-card"><div class="source-name">{source["document_name"]}</div><div class="source-score">Relevance {source["score"]:.3f}</div><div class="source-text">{source["text"][:420]}...</div></div>',
                        unsafe_allow_html=True,
                    )
            elif st.session_state["use_hybrid_search"]:
                st.info("No matching passages were found in the local FAISS index.")


# Main execution
if __name__ == "__main__":
    render_chatbot_page()
