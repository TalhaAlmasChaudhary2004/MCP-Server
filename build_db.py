from ingest import load_and_split_pdfs
from embeddings import create_vectorstore

# Step 1: Load PDFs and split into chunks
chunks = load_and_split_pdfs()

# Step 2: Create vector store
create_vectorstore(chunks)