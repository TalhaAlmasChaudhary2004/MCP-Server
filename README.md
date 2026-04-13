# Document QA MCP Server

This project implements a Model Context Protocol (MCP) server for Document Question Answering using FastMCP. It processes PDF files, chunks the text, creates a vector database using HuggingFace embeddings (`sentence-transformers`), and answers incoming queries.

## Project Structure

- `data/`: Directory containing your source PDF files.
- `ingest.py`: Loads PDF files from the `data/` directory and splits them into smaller text chunks.
- `embeddings.py`: Handles initializing the embedding model and creates the Chroma vector store (`vector_store/`).
- `build_db.py`: The script to build the vector database using `ingest.py` and `embeddings.py`.
- `retriever.py`: Sets up retrieval functionality from the stored Chroma database.
- `qa.py`: Uses the retriever to fetch relevant document chunks.
- `mcpserver.py`: Exposes a `query_documents` MCP tool over FastMCP.
- `requirements.txt`: Python package requirements.

## Prerequisites & Setup

### 1. Create a Virtual Environment

It is recommended to use **Python 3.13** for this project. Navigate to your project directory and create a virtual environment:

**Windows:**
```shell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```shell
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Once your virtual environment is active, install all the prerequisite libraries from `requirements.txt`:

```shell
pip install -r requirements.txt
```

*(Note: ensure you have libraries like `sentence-transformers`, `chromadb`, `pymupdf`, `mcp`, and `langchain` suite correctly installed).*

## Running the Application

To properly configure and start the application, follow the scripts in this exact order:

1. **`ingest.py` & `embeddings.py`**: These modules contain the logic to load PDFs and define the Vector Store format.
2. **`build_db.py`**: Run this script first to execute the ingestion and embedding processes and create your persistent vector database.
   ```shell
   python build_db.py
   ```
3. **`retriever.py` & `qa.py`**: These scripts test the logic for fetching relevant document chunks and aggregating the retrieved information.
4. **`mcpserver.py`**: Finally, after the vector database is fully populated, start the FastMCP server:
   ```shell
   python mcpserver.py
   ```

You are now successfully running the `document-qa-mcp-server` and can start querying your ingested datasets via the Model Context Protocol!

