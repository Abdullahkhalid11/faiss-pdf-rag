"""
Configuration module for FAISS PDF RAG.
Reads from environment variables (or .env in development).
"""
import os
from dotenv import load_dotenv

# Load .env only in development
if os.getenv("ENVIRONMENT") != "production":
    load_dotenv()

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
INDEX_DIR = os.getenv("INDEX_DIR", "./faiss_index")

# FastAPI Configuration
API_KEY = os.getenv("API_KEY")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
PORT = int(os.getenv("PORT", "8000"))

# Validation for production
def validate_config():
    """Validate critical config in production."""
    if ENVIRONMENT == "production":
        assert GROQ_API_KEY, "GROQ_API_KEY must be set in production"
        assert API_KEY, "API_KEY must be set in production"
        assert API_KEY != "my_production_secret_key_12345_change_this", "API_KEY cannot be default value"
        print("✅ Production config validated")

validate_config()