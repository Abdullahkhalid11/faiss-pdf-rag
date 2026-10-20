"""
app.py — FastAPI server for RAG service.
Routes:
  POST /upload — Upload PDF(s) and rebuild FAISS index
  POST /chat — Query the RAG chain
  GET /health — Health check
 
Usage: uvicorn app:app --reload
"""
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
 
from src.ingest import ingest
from src.retrieve import RAGChain
from src.config import INDEX_DIR
 
app = FastAPI(title="FAISS PDF RAG", version="1.0")
 
# Initialize RAG chain (lazy load on first request)
rag_chain = None
 
 
def get_rag_chain():
    """Load RAG chain once, reuse for all queries."""
    global rag_chain
    if rag_chain is None:
        index_path = Path(INDEX_DIR)
        if not index_path.exists():
            raise HTTPException(status_code=400, detail="No FAISS index found. Upload PDFs first.")
        rag_chain = RAGChain(INDEX_DIR)
    return rag_chain
 
 
class QueryRequest(BaseModel):
    question: str
 
 
class QueryResponse(BaseModel):
    answer: str
    sources_count: int
 
 
@app.get("/health")
async def health():
    """Health check endpoint."""
    index_exists = Path(INDEX_DIR).exists()
    return {
        "status": "ok",
        "index_ready": index_exists,
        "index_path": str(INDEX_DIR)
    }
 
 
@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """
    Upload PDF files and rebuild FAISS index.
 
    Returns:
      - files_uploaded: number of PDFs processed
      - chunks_created: total chunks in new index
      - index_path: path to FAISS index
    """
    pdf_dir = Path("./pdfs")
    pdf_dir.mkdir(exist_ok=True)
 
    # Clear old PDFs
    for old_pdf in pdf_dir.glob("*.pdf"):
        old_pdf.unlink()
 
    uploaded_count = 0
 
    # Save uploaded files
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
 
        file_path = pdf_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        uploaded_count += 1
 
    if uploaded_count == 0:
        raise HTTPException(status_code=400, detail="No PDF files uploaded")
 
    # Rebuild index
    try:
        docs = ingest(str(pdf_dir), INDEX_DIR)
 
        # Reset RAG chain so it reloads the new index
        global rag_chain
        rag_chain = None
 
        return {
            "status": "success",
            "files_uploaded": uploaded_count,
            "index_path": str(INDEX_DIR),
            "message": f"Uploaded {uploaded_count} PDF(s) and rebuilt FAISS index"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
 
 
@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Query the RAG chain.
 
    Request body:
      {
        "question": "your question here"
      }
 
    Returns:
      {
        "answer": "response from Groq",
        "sources_count": number of documents retrieved
      }
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
 
    try:
        chain = get_rag_chain()
        result = chain.query(request.question)
 
        return QueryResponse(
            answer=result["answer"],
            sources_count=len(result["sources"])
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="FAISS index not found. Upload PDFs first via /upload")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)