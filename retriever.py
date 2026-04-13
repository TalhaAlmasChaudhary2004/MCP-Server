from langchain_community.vectorstores import Chroma
from embeddings import DB_PATH, get_embedding_model

"""
retriever.py

"""

# This function is called by the MCP server (via `_get_retriever_cached`) to
# construct a retriever object backed by the persisted Chroma DB.
# The retriever can then be used to fetch the most relevant chunks for any
# given question.

def get_retriever():

    """
    Load the persisted vector DB and return a retriever.

    Returns:
        A LangChain retriever object. When called with a query string, it will:
        - embed the query using `get_embedding_model()`
        - perform a similarity search over the Chroma vector store at DB_PATH
        - return the top-k (here, k=3) most relevant chunks.
    """
    
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=get_embedding_model(),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever