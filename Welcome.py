import logging
import os

import streamlit as st

from src.utils import setup_logging

# Initialize logger
setup_logging()  # Set up logging configuration
logger = logging.getLogger(__name__)

# Set page config with title, icon, and layout
st.set_page_config(
    page_title="Panda's Chatbot", page_icon="🤖"
)


# Custom CSS to style the page and sidebar
def apply_custom_css() -> None:
    """Applies custom CSS styling to the Streamlit page and sidebar."""
    st.markdown(
        """
        <style>
            :root { --canvas: #201e1b; --ink: #faf9f5; --body: #d6d0c7; --muted: #a09d96; --hairline: #3a3631; --dark: #181715; --dark-soft: #302c27; --coral: #cc785c; --coral-dark: #a9583e; --teal: #5db8a6; }
            .stApp { background: var(--canvas); color: var(--body); }
            .block-container { max-width: 1180px; padding: 3rem clamp(1rem, 5vw, 5rem) 5rem; }
            [data-testid="stSidebar"] { background: var(--dark); }
            [data-testid="stSidebar"] * { color: #faf9f5; }
            h1, h2, h3, h4 { color: var(--ink); font-family: Georgia, "Times New Roman", serif; font-weight: 400; letter-spacing: 0; }
            h1 { font-size: clamp(2.8rem, 7vw, 5.8rem); line-height: .98; max-width: 12ch; margin: 1rem 0; }
            h2 { font-size: 1.7rem; }
            .kicker { color: var(--coral-dark); font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
            .lead { max-width: 38rem; color: var(--muted); font-size: 1.08rem; line-height: 1.65; }
            .welcome-rule { width: 5rem; height: 3px; background: var(--coral); margin: 2rem 0; }
            .signal-row { display: flex; gap: .6rem; flex-wrap: wrap; margin: 2rem 0 3rem; }
            .signal { border: 1px solid var(--hairline); border-radius: 999px; padding: .45rem .75rem; background: #2a2723; color: var(--muted); font-size: .78rem; }
            .signal.ready { border-color: #3f756a; background: #263b36; color: #9ed8cb; }
            .route-label { color: var(--muted); font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .7rem; }
            .route-copy { color: var(--body); line-height: 1.55; margin-bottom: 1.2rem; }
            [data-testid="stSidebar"] { border-right: 1px solid var(--dark-soft); }
            .stButton button { background: var(--coral); color: #fffdf9; border: 1px solid var(--coral-dark); border-radius: 8px; min-height: 44px; }
            .stButton button:hover { background: var(--coral-dark); color: #fffdf9; }
            .footer-text { color: #c9c5bc; font-size: .82rem; text-align: center; margin-top: 2rem; }
            @media (max-width: 767px) { .block-container { padding: 1.5rem 1rem 4rem; } h1 { font-size: 3.1rem; } .lead { font-size: 1rem; } }
            @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logger.info("Applied custom CSS styling.")


# Function to display logo or placeholder
def display_logo(logo_path: str) -> None:
    """Displays the logo in the sidebar or a placeholder if the logo is not found.

    Args:
        logo_path (str): The file path for the logo image.
    """
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=220)
        logger.info("Logo displayed.")
    else:
        st.sidebar.markdown("### Logo Placeholder")
        logger.warning("Logo not found, displaying placeholder.")


# Function to display main content
def display_main_content() -> None:
    """Displays the main welcome content on the page."""
    st.markdown('<div class="kicker">Panda\'s Chatbot / local workspace</div>', unsafe_allow_html=True)
    st.title("A clearer way to ask your documents.")
    st.markdown('<div class="welcome-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="lead">Upload PDFs, search their passages locally, and ask questions with the evidence kept close to the answer.</p>
        """
        , unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="signal-row"><span class="signal ready">FAISS ready</span><span class="signal">Ollama stays local</span><span class="signal">PDF-first answers</span></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="route-label">Start here</div>', unsafe_allow_html=True)
        st.markdown('<div class="route-copy">Add a document to the workspace. Its text is chunked, embedded, and saved locally for retrieval.</div>', unsafe_allow_html=True)
        st.page_link("pages/2_📄_Upload_Documents.py", label="Upload documents", icon=":material/upload_file:")
    with right:
        st.markdown('<div class="route-label">Already uploaded?</div>', unsafe_allow_html=True)
        st.markdown('<div class="route-copy">Go straight to the assistant and ask about dates, requirements, roles, or passages in your PDFs.</div>', unsafe_allow_html=True)
        st.page_link("pages/1_🤖_Chatbot.py", label="Open the chatbot", icon=":material/chat:")
    logger.info("Displayed main welcome content.")


# Function to display sidebar content
def display_sidebar_content() -> None:
    """Displays headers and footer content in the sidebar."""
    st.sidebar.markdown(
        "<h2 style='text-align: center;'>Panda's Chatbot</h2>", unsafe_allow_html=True
    )
    st.sidebar.markdown(
        "<h4 style='text-align: center;'>Your Document Assistant</h4>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div class="footer-text">
            © 2024 Panda's Chatbot
        </div>
        """,
        unsafe_allow_html=True,
    )
    logger.info("Displayed sidebar content.")


# Main execution
if __name__ == "__main__":
    apply_custom_css()
    display_logo("images/logo.png.png")
    display_sidebar_content()
    display_main_content()
