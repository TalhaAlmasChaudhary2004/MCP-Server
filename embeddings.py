import os
from functools import lru_cache
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

"""
embeddings.py

"""


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "vector_store")



@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create a HuggingFace sentence-transformers model.

    The first time this function is called in a process, it will:
    - download/load the model "sentence-transformers/all-MiniLM-L6-v2" if
      necessary,
    - construct a `HuggingFaceEmbeddings` wrapper around it,
    - cache that instance.

    On subsequent calls, the cached instance is returned immediately, so we:
    - avoid reloading the model into memory,
    - keep MCP server startup and query handling fast.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



def create_vectorstore(chunks):
    """
    Create and persist a Chroma vector database from a list of text chunks.

    Args:
        chunks: Iterable of LangChain `Document` objects, typically produced by
                `load_and_split_pdfs` in `ingest.py`.

    Steps:
        1. Ensure the target directory for the DB exists (`DB_PATH`).
        2. Use `Chroma.from_documents` to:
           - embed each chunk using the shared embedding model,
           - store embeddings + metadata in a Chroma index on disk.
        3. Call `persist()` so the DB is fully written to `DB_PATH` and can be
           reloaded later by `retriever.py`.

    """
   
    os.makedirs(DB_PATH, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=DB_PATH,
    )

   
    vectorstore.persist()

    return vectorstore