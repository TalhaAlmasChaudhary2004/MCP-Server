from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI 
import os, json, time
from pathlib import Path


"""
qa.py
-----

This module implements a simple retrieval-based question answering system.

Functionality:
- Takes a user query (question)
- Uses a retriever to fetch relevant document chunks
- Extracts text content from retrieved documents
- Returns combined text as the answer along with source metadata

"""



def retrieve_answer(question, retriever):
    """
    Retrieve relevant text chunks from retriever 
    """

    docs = retriever.invoke(question)
    

    # Extract text (limit to 500 chars per chunk)
    answer_parts = [(doc.page_content or "")[:500] for doc in docs]
    answer = "\n\n".join([part for part in answer_parts if part.strip()])

    # Extract sources (metadata)
    sources = []
    for doc in docs:
        sources.append({
            "document": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        })

    return answer, sources