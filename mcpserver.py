from mcp.server.fastmcp import FastMCP
from qa import retrieve_answer
from retriever import get_retriever 
import sys
"""
mcpserver.py

"""

_retriever = None


def _get_retriever_cached():
    """
    Return a singleton retriever instance.
    create the retriever and store it in `_retriever`.
    return the cached retriever immediately.
    """

    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever



# MCP server setup

mcp = FastMCP("document-qa-mcp-server")


# MCP tool: query_documents

@mcp.tool()
def query_documents(question: str) -> dict:

    """
    MCP tool: query_documents

    Input:
    - question: the user's question (string)

    Output:
    - dict with:
      - question: echoed back
      - answer: generated answer text
      - sources: list of {document, page} pairs from retrieved chunks
    """

    retriever = _get_retriever_cached()
    answer, sources = retrieve_answer(question, retriever)

    return {
        "question": question,
        "check return":"hello",
        "answer": answer,
        "sources": sources
    }



# Start MCP server 

if __name__ == "__main__":
    print("Starting MCP server...", file=sys.stderr)
    mcp.run()