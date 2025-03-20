# Rag-Gemma3

## Overview

Rag-Gemma3 is a Retrieval-Augmented Generation (RAG) system powered by the Gemma-3 model. It allows users to retrieve information from pre-processed or uploaded documents and generate responses using a large language model (LLM). The project is built using Python and leverages LangChain, Pinecone, Hugging Face models, and Gradio for the UI.

## Features

- **Retrieval-Augmented Generation (RAG)**: Fetches relevant documents and generates responses using an LLM.
- **Multiple Document Processing Options**: Supports both pre-processed and user-uploaded documents.
- **Gradio UI**: Provides a simple and interactive chatbot interface.
- **Pinecone Vector Database**: Efficiently stores and retrieves document embeddings.
- **Hugging Face Integration**: Uses `Gemma-3` for text generation and `BAAI/bge-small-en-v1.5` for embeddings.
- **Flask API**: Serves the LLM and embedding models via API endpoints.

## Project Structure

```
othmansamih-rag-gemma3/
├── requirements.txt                # List of dependencies
├── .env_example                    # Example environment variables
├── configs/                         # Configuration files
│   └── app_config.yaml              # Main configuration file
├── data/                            # Directory for storing documents
│   └── documents/                    # Pre-processed document storage
├── src/                             # Main source code directory
│   ├── app.py                        # Gradio UI application
│   ├── process_documents_manually.py # Script to process and store documents in Pinecone
│   ├── serve_llm_and_embedding_models.py # API server for LLM and embeddings
│   ├── utils/                        # Utility scripts
│   │   ├── chatbot.py                 # Handles chatbot interactions
│   │   ├── clean_chatbot.py           # Cleans chatbot data and uploaded files
│   │   ├── custom_api.py              # Custom API for embeddings and LLM
│   │   ├── load_app_config.py         # Loads YAML configuration
│   │   ├── prepare_vectordb.py        # Prepares and manages Pinecone vector database
│   │   ├── ui_settings.py             # Handles UI-related settings
│   │   ├── upload_document.py         # Manages document uploads
│   │   ├── utilities.py               # Helper functions for various tasks
```

## Installation

### Prerequisites

- Python 3.9+
- `pip` package manager
- Pinecone API Key
- Hugging Face API Token

### Setup

1. **Clone the repository:**
    
    ```bash
    git clone https://github.com/<yourusername>/Rag-Gemma3.git
    cd Rag-Gemma3
    ```
    
2. **Install dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. **Set up environment variables:**

    - Copy `.env_example` to `.env` and update it with your API keys.    

    ```bash
    PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
    HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACEHUB_API_TOKEN"
    ```
    

## Configuration

Edit `configs/app_config.yaml` to configure paths, model parameters, and API endpoints.

```yaml
directories:
  documents_dir: "./data/documents"
  uploaded_documents_dir: "./data/uploaded_documents"

vectordb_config:
  text_splitter:
    chunk_size: 1500
    chunk_overlap: 250
  index:
    index_name: "rag-gemma3"
    cloud: "aws"
    region: "us-east-1"
  retrieved_docs:
    k: 2

embeddings:
  embed_model_id: "BAAI/bge-small-en-v1.5"

llm:
  gen_model_id: "google/gemma-3-1b-it"
  temperature: 0.1
  device_map: "cpu"
  max_new_tokens: 256
  system_prompt: >
    You are a helpful AI assistant. Respond to the user's prompt based on the retrieved documents.
    If the retrieved documents are not relevant, reply with: 'Sorry, I don't have enough information about this.'

```

## Usage

### 1. Start the LLM and Embedding API

```bash
python src/serve_llm_and_embedding_models.py
```

- This will start a Flask server on `http://127.0.0.1:5000`

### 2. Process Documents (Pre-processing)

```bash
python src/process_documents_manually.py
```

- This will load documents from `data/documents`, split them into chunks, generate embeddings, and store them in Pinecone.

### 3. Run the Chatbot UI

```bash
gradio src/app.py
```

- Open `http://127.0.0.1:7860` in a browser to interact with the chatbot.
