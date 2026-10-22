"""
ingest.py — Load PDFs, chunk, embed, save FAISS index.
Usage: python -m src.ingest --pdf-dir ./pdfs --index-dir ./faiss_index
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, INDEX_DIR

load_dotenv()


def load_pdfs(pdf_dir: str) -> list:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    paths = list(pdf_dir.glob("*.pdf"))
    if not paths:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    docs = []
    for p in paths:
        loader = PyPDFLoader(str(p))
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} pages from {len(paths)} PDF(s)")
    return docs


def chunk_docs(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_index(chunks: list, index_dir: str) -> FAISS:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(index_dir))
    print(f"FAISS index saved to {index_dir} ({store.index.ntotal} vectors)")
    return store


def ingest(pdf_dir: str = "./pdfs", index_dir: str = INDEX_DIR) -> FAISS:
    docs = load_pdfs(pdf_dir)
    chunks = chunk_docs(docs)
    return build_index(chunks, index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS index")
    parser.add_argument("--pdf-dir", default="./pdfs")
    parser.add_argument("--index-dir", default=INDEX_DIR)
    args = parser.parse_args()
    ingest(args.pdf_dir, args.index_dir)