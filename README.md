
# CiteCraft Backend — RAG API

AI-powered document research API built with FastAPI, FAISS, and LLMs.

**[API Docs](https://faiss-pdf-rag.onrender.com/docs)** | **[Frontend](https://github.com/Abdullahkhalid11/faiss-rag-ui)** | **[Built by Abdullah Khalid](https://github.com/Abdullahkhalid11)**

---

## What is CiteCraft?

CiteCraft is a retrieval-augmented generation (RAG) API that turns PDF documents into a searchable knowledge base. Every answer is grounded in document sources—no hallucinations, just facts.

### Key Features

✅ **PDF Ingestion & Indexing** — Automatic chunking and embedding
✅ **Vector Search** — FAISS-powered semantic retrieval
✅ **LLM Integration** — Groq API for answer generation
✅ **Source Traceability** — Every answer includes document chunks
✅ **API Authentication** — X-API-Key header validation
✅ **Production Ready** — Error handling, logging, CORS, rate limiting
✅ **Fully Deployed** — Running on Render

---

## Architecture

PDF Upload
↓
PyPDF Loader
↓
RecursiveCharacterTextSplitter (chunks)
↓
Jina Embeddings v2
↓
FAISS Vector Store
↓
Similarity Search (top-k retrieval)
↓
LangChain + Groq LLM
↓
Answer with Source Citations

---

## Tech Stack

| Component                | Technology                 |
| ------------------------ | -------------------------- |
| **Framework**      | FastAPI 0.115.4            |
| **Language**       | Python 3.12                |
| **Vector DB**      | FAISS (faiss-cpu 1.9.0)    |
| **Embeddings**     | Jina Embeddings v2         |
| **LLM**            | Groq (openai/gpt-oss-120b) |
| **RAG**            | LangChain 0.3.7            |
| **PDF Processing** | PyPDF 4.2.0                |
| **Server**         | Uvicorn 0.30.1             |
| **Validation**     | Pydantic 2.7.4             |
| **Deployment**     | Docker + Render            |

---

## Local Development

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
git clone https://github.com/Abdullahkhalid11/faiss-pdf-rag.git
cd faiss-pdf-rag

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with:
# - GROQ_API_KEY (from console.groq.com/keys)
# - JINA_API_KEY (from jina.ai)
# - API_KEY (random secret)

# Run server
uvicorn app:app --reload
```

Access API docs: http://localhost:8000/docs

---

## API Reference

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "ok",
  "index_ready": true,
  "index_path": "./faiss_index"
}
```

---

### Upload & Index PDFs

```http
POST /upload
X-API-Key: your_api_key
Content-Type: multipart/form-data

files: [pdf1.pdf, pdf2.pdf, ...]
```

**Response:**

```json
{
  "status": "success",
  "files_uploaded": 2,
  "message": "Uploaded 2 PDF(s) and rebuilt FAISS index"
}
```

---

### Query RAG System

```http
POST /chat
X-API-Key: your_api_key
Content-Type: application/json

{
  "question": "What is this document about?"
}
```

**Response:**

```json
{
  "answer": "The document discusses...",
  "sources_count": 3,
  "sources": [
    {
      "content": "Relevant chunk from page 1...",
      "page": 0
    },
    {
      "content": "Relevant chunk from page 2...",
      "page": 1
    }
  ]
}
```

---

## Example Requests

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Upload PDFs
curl -X POST http://localhost:8000/upload \
  -H "X-API-Key: your_api_key" \
  -F "files=@document.pdf"

# Query
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main topics?"}'
```

### Using Python

```python
import requests

API_URL = "http://localhost:8000"
API_KEY = "your_api_key"
headers = {"X-API-Key": API_KEY}

# Upload
with open("document.pdf", "rb") as f:
    files = {"files": f}
    response = requests.post(
        f"{API_URL}/upload",
        headers=headers,
        files=files
    )
    print(response.json())

# Query
response = requests.post(
    f"{API_URL}/chat",
    headers=headers,
    json={"question": "What is this about?"}
)
print(response.json())
```

---

## Project Structure

faiss-pdf-rag/
├── app.py # FastAPI server
├── src/
│ ├── init.py
│ ├── auth.py # API key verification
│ ├── config.py # Environment configuration
│ ├── ingest.py # PDF → FAISS pipeline
│ └── retrieve.py # RAG chain (search + LLM)
├── faiss_index/ # Vector store (pre-built)
│ ├── index.faiss
│ └── index.pkl
├── pdfs/ # PDF documents (uploaded)
├── Dockerfile # Production container
├── requirements.txt # Python dependencies
├── .env.example # Environment template
└── README.md

---

## Deployment

### Using Render

1. **Push to GitHub**

```bash
   git push origin main
```

2. **Create Render Web Service**

   - Connect GitHub repository
   - Select `main` branch
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port 8000`
3. **Set Environment Variables**

   - `GROQ_API_KEY` — Your Groq API key
   - `JINA_API_KEY` — Your Jina API key
   - `API_KEY` — Random secret (e.g., `supersecretkey123`)
   - `ENVIRONMENT` — `production`
4. **Deploy**

   - Click Deploy
   - Wait 5 minutes for build to complete

---

### Using Docker Locally

```bash
docker build -t citecraft-api .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -e JINA_API_KEY=your_key \
  -e API_KEY=your_api_key \
  -e ENVIRONMENT=production \
  citecraft-api
```

---

## Performance Benchmarks

| Metric                      | Result                |
| --------------------------- | --------------------- |
| Health check                | <50ms                 |
| PDF upload (10 MB)          | ~30 seconds           |
| Similarity search           | <500ms                |
| LLM response (Groq)         | 1-2 seconds           |
| **Total Q&A latency** | **2-3 seconds** |

---

## Configuration

### Environment Variables

```env
# LLM Provider
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-120b

# Embeddings
JINA_API_KEY=jina_...
EMBED_MODEL=jina-embeddings-v2-base-en

# RAG Pipeline
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
INDEX_DIR=./faiss_index

# Security
API_KEY=your_random_secret_key_change_this
ENVIRONMENT=development

# Server
PORT=8000
```

---

## Limitations & Future Work

### Current Limitations

- Single-turn conversations (no memory between queries)
- English language support only
- Scanned PDFs not supported (no OCR)

### Planned Improvements

#### 🔐 AI Security

- [ ] Prompt injection detection & mitigation
- [ ] LLM guardrails (content filtering, output validation)
- [ ] Input/output encoding safety

#### 🏗️ System Design & Scalability

- [ ] Hybrid RAG with semantic reranking
- [ ] LLM response caching (Redis)
- [ ] Load balancing & horizontal scaling
- [ ] Multi-tenant architecture with data isolation

#### 📊 Evaluation & Observability

- [ ] RAGAS metrics (semantic similarity, answer relevance)
- [ ] LangSmith tracing & debugging
- [ ] Token usage & cost monitoring
- [ ] Query latency analytics

#### 🚀 Production Hardening

- [ ] Circuit breakers for LLM API failures
- [ ] Request queuing & backpressure handling
- [ ] Batch PDF processing
- [ ] Advanced error recovery

#### 🔄 Advanced Features

- [ ] Multi-turn conversation memory
- [ ] Document metadata filtering (date range, author)
- [ ] OCR support for scanned PDFs
- [ ] Custom embedding model fine-tuning
- [ ] Knowledge graph integration

---

## What This Project Demonstrates

✅ **End-to-End RAG:** PDF parsing → chunking → embedding → retrieval → LLM
✅ **Production API:** Authentication, error handling, logging, CORS
✅ **Vector Search:** FAISS indexing and semantic similarity
✅ **LLM Integration:** Prompt engineering with source context
✅ **Full-Stack Deployment:** Backend on Render + frontend on Vercel
✅ **Source Transparency:** Answer traceability with document chunks

**⚠️ Not included:** Enterprise security, distributed systems, advanced caching
**Why?** Scope — these are discussed based on client requirements.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'langchain_huggingface'`**

```bash
pip install langchain-huggingface
```

**`GROQ_API_KEY not set in production`**

- Verify `GROQ_API_KEY` is set in Render environment variables
- Restart the service

**`FAISS index not found`**

- Upload a PDF first via `/upload` endpoint
- Index will be created automatically

**`Connection refused` (localhost:8000)**

- Check if uvicorn is running: `uvicorn app:app --reload`
- Verify port 8000 is not in use

---

## Security Notes

- 🔒 All endpoints except `/health` require `X-API-Key` authentication
- 🔒 API keys are validated on every request
- 🔒 PDFs are processed in isolation (per upload request)
- 🔒 `.env` file is in `.gitignore` (never commit secrets)
- ⚠️ For production, use HTTPS and rotate API keys regularly

---

## About the Developer

**Abdullah Khalid** — AI Engineer

I build AI applications across computer vision, machine learning, LLM applications, and automation. CiteCraft demonstrates end-to-end RAG architecture: from PDF parsing through semantic search to LLM-powered Q&A.

**Skills:** Python • FastAPI • RAG • Vector Search • LLMs • LangChain • FAISS • System Design

**Links:** [GitHub](https://github.com/Abdullahkhalid11) | [Email](mailto:msabdullah2024@gmail.com)

---

## License

© 2026 Abdullah Khalid. All rights reserved.

CiteCraft is an independent portfolio project. Not affiliated with Groq, Jina, LangChain, or other third-party providers.

---

## Questions?

- 📧 Email: [msabdullah2024@gmail.com](mailto:msabdullah2024@gmail.com)
- 💬 Open an issue: [GitHub Issues](https://github.com/Abdullahkhalid11/faiss-pdf-rag/issues)
- 🔗 View frontend: [GitHub](https://github.com/Abdullahkhalid11/faiss-rag-ui)
