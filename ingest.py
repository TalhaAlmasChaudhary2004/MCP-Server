import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

"""
This function loads all PDF files from a specified directory and prepares them
for downstream .

Steps:
1. Iterate through all files in the DATA_PATH directory.
2. Filter and process only PDF files.
3. Load each PDF using PyMuPDFLoader, extracting content page by page.
4. Store all pages into a single documents list.
5. Use RecursiveCharacterTextSplitter to break large text into smaller chunks.
6. Apply chunk_size and chunk_overlap to preserve context across chunks.
7. Return the final list of text chunks.

"""



DATA_PATH = "data/"


def load_and_split_pdfs(chunk_size=500, chunk_overlap=50):
    """
    Load PDFs and split into chunks for embedding
    """
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_PATH, file))
            docs = loader.load()
            documents.extend(docs)

    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")
    return chunks