📝 Build Your Local RAG System with LLMs

Welcome to Local PDF Chatbot — a local Retrieval-Augmented Generation (RAG) system that allows you to upload PDF documents and ask questions about their contents using local AI models.

The system uses FAISS for vector similarity search and Ollama for local embeddings and LLM inference, keeping your documents and queries on your own machine.

🌟 Key Features

🔒 Privacy-Friendly Document Search: Search through personal documents without uploading them to the cloud.

🔍 Local Vector Search with FAISS: Stores document embeddings in a persistent FAISS index without requiring a separate database container.

🧠 Local Ollama Inference: Uses Ollama to generate embeddings and responses locally.

📄 PDF Document Processing: Extracts and chunks text from uploaded PDF documents.

💬 Context-Aware Responses: Retrieves relevant passages from your documents before generating an answer.

⚡ Simple Local Deployment: Designed to run locally with a straightforward Python and Streamlit setup.

🏗️ Architecture

The system follows a Retrieval-Augmented Generation (RAG) pipeline. Uploaded documents are processed locally, converted into embeddings, stored in a persistent FAISS index, and retrieved when a user asks a question.

PDF upload
    |
    v
Text extraction and chunking
    |
    v
Ollama nomic-embed-text
    |
    v
Persistent FAISS index
    |
    v
Question embedding and similarity search
    |
    v
Retrieved passages
    |
    v
Ollama TinyLlama response



🔄 How It Works

1. Upload Documents

PDF documents are uploaded through the Streamlit interface.

2. Extract and Chunk Text

The application extracts text from the uploaded PDF and divides it into smaller chunks that can be efficiently embedded and retrieved.

3. Generate Embeddings

The text chunks are converted into vector embeddings using Ollama's:

nomic-embed-text

4. Store Embeddings

The generated embeddings are stored in a persistent FAISS index for efficient similarity search.

5. Process the Question

When a user asks a question, the question is converted into an embedding using the same embedding model.

6. Retrieve Relevant Passages

FAISS performs similarity search against the stored document embeddings and retrieves the most relevant passages.

7. Generate the Response

The retrieved passages are provided as context to the local LLM, which generates a grounded response.

🛠️ Tech Stack

Component

Technology

Interface

Streamlit

Programming Language

Python

Document Processing

PyMuPDF

Vector Database

FAISS

Embedding Model

Ollama nomic-embed-text

Language Model

Ollama TinyLlama

OCR

Tesseract

Package Management

pyproject.toml / pip

📁 Project Structure

beginner-local-rag-system/
│
├── Welcome.py
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── mypy.ini
├── .gitignore
├── rag-pipeline-architecture.png
│
├── pages/
│   ├── 1_🧖_Chatbot.py
│   └── 2_📄_Upload_Documents.py
│
└── src/
    ├── __init__.py
    ├── chat.py
    ├── constants.py
    ├── embeddings.py
    ├── faiss_store.py
    ├── ingestion.py
    ├── ocr.py
    └── utils.py

⚙️ Requirements

Before running the application, make sure you have:

Python 3.11 or a compatible Python version

Ollama

nomic-embed-text

TinyLlama

Tesseract OCR if OCR functionality is required

🚀 Installation

1. Clone the repository

git clone https://github.com/kartikey009/Local_document_assistant-.git
cd Local_document_assistant-

2. Create a virtual environment

Windows PowerShell:

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

4. Install Ollama Models

Make sure Ollama is installed and running, then pull the required models:

ollama pull nomic-embed-text
ollama pull tinyllama

5. Run the application

streamlit run Welcome.py

The Streamlit interface should open in your browser.

💡 Usage

Start the Streamlit application.

Open the Upload Documents page.

Upload one or more PDF documents.

Allow the application to process and index the documents.

Open the Chatbot page.

Ask questions about the uploaded documents.

The system retrieves relevant passages and uses them as context for the local LLM response.

🔐 Privacy

This project is designed around local document processing.

Documents are processed on the deployment machine and are not intentionally uploaded to a cloud-based AI service. Embeddings are generated using Ollama running locally, while FAISS provides local vector similarity search.

⚠️ Limitations

Ollama must be installed and running on the host machine.

Response speed depends on available CPU, GPU, and system memory.

Large documents may require more processing time and memory.

Uploaded files are local to the deployment machine.

There is currently no built-in authentication.

Scanned image-only PDFs may require OCR processing.

Local LLM performance depends on the selected model and available hardware.

Retrieval quality depends on document chunking, embedding quality, and similarity search.

🔮 Future Improvements

Potential improvements include:

Support for additional document formats.

Improved OCR support for scanned PDFs.

Better chunking and retrieval strategies.

Metadata-aware document retrieval.

Conversation history and multi-turn context.

Source citations for retrieved passages.

Authentication and user management.

Support for multiple embedding and generation models.

Improved UI and document management.

Evaluation metrics for retrieval and response quality.

📌 Project Status

This project is a local RAG system built for learning, experimentation, and demonstrating the fundamentals of document retrieval and local LLM integration.
