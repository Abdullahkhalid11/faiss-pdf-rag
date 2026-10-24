"""
app.py — FastAPI server for RAG service.
"""
import os
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.ingest import ingest
from src.retrieve import RAGChain
from src.config import INDEX_DIR
from src.auth import verify_api_key
import uvicorn

app = FastAPI(title="FAISS PDF RAG", version="1.0")

# ADD THIS CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for demo; restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add explicit CORS preflight handlers
@app.options("/upload")
@app.options("/chat")
async def preflight():
    """Handle CORS preflight requests."""
    return {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG chain
rag_chain = None

def get_rag_chain():
    """Load RAG chain once, reuse for all queries."""
    global rag_chain
    if rag_chain is None:
        index_path = Path(INDEX_DIR)
        if not index_path.exists():
            raise FileNotFoundError(f"No FAISS index found.")
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
async def upload_pdfs(
    files: list[UploadFile] = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Upload PDF files and rebuild FAISS index."""
    logger.info(f"Upload started - Files: {len(files)}")
    
    pdf_dir = Path("./pdfs")
    pdf_dir.mkdir(exist_ok=True)
    
    for old_pdf in pdf_dir.glob("*.pdf"):
        old_pdf.unlink()
    
    uploaded_count = 0
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        file_path = pdf_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        uploaded_count += 1
        logger.info(f"Saved: {file.filename}")
    
    if uploaded_count == 0:
        raise HTTPException(status_code=400, detail="No PDF files uploaded")
    
    try:
        docs = ingest(str(pdf_dir), INDEX_DIR)
        global rag_chain
        rag_chain = None
        logger.info(f"Index rebuilt - {uploaded_count} PDFs processed")
        
        return {
            "status": "success",
            "files_uploaded": uploaded_count,
            "message": f"Uploaded {uploaded_count} PDF(s) and rebuilt FAISS index"
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/chat", response_model=QueryResponse)
async def chat(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Query the RAG chain with API key authentication."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    logger.info(f"Query received - Question: {request.question[:50]}...")
    
    try:
        chain = get_rag_chain()
        result = chain.query(request.question)
        logger.info(f"Query successful - Sources: {len(result['sources'])}")
        
        return QueryResponse(
            answer=result["answer"],
            sources_count=len(result["sources"])
        )
    except FileNotFoundError:
        logger.error("FAISS index not found")
        raise HTTPException(status_code=400, detail="FAISS index not found. Upload PDFs first via /upload")
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)