# FAISS PDF RAG System

A Retrieval-Augmented Generation (RAG) system that allows querying PDF documents using FAISS, LangChain, and Groq LLM.

---

## Features

- Load and process multiple PDF files
- Text chunking and embeddings using Sentence Transformers
- Vector storage using FAISS
- Semantic search with LLM-based answers
- Environment variable based configuration

---

## Tech Stack

- Python 3.10+
- LangChain
- FAISS (CPU)
- Sentence Transformers
- Groq LLM
- PyPDF
- Jupyter Notebook

---

## Project Structure

```
faiss-pdf-rag/
├── faiss_pdf_rag.ipynb      # Main Jupyter notebook
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files
├── pdfs/                   # Place your PDF files here
├── faiss_index_pdfs/       # Generated FAISS index
└── .env                    # Environment variables (not committed)
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Abdullahkhalid11/faiss-pdf-rag.git
cd faiss-pdf-rag
```

---

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

### 5. Groq API Key Setup

This project uses Groq LLM. You must provide your own API key.

1. Get an API key from:
   https://console.groq.com/keys

2. Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_api_key_here

---

### 6. Run the Project

```bash
jupyter notebook
```

Open:

```
faiss_pdf_rag.ipynb
```

---

## How It Works

1. PDF files are loaded from the `pdfs/` directory
2. Text is split into chunks
3. Chunks are embedded using Sentence Transformers
4. Embeddings are stored in FAISS
5. Relevant chunks are retrieved based on user query
6. Groq LLM generates the final answer

---

## Important Notes

- Do not commit `.env` or `venv/`
- FAISS index is generated automatically
- Add your PDF files inside the `pdfs/` directory

---

## Author

Abdullah  
Senior AI Application Developer  

Focus areas:
- RAG Systems
- LLM Applications
- Computer Vision
- Edge AI

---

## Support

If you find this project useful, please give it a star on GitHub.
