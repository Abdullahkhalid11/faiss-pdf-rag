"""
Authentication module for API security.
"""
import os
from fastapi import Header, HTTPException

API_KEY = os.getenv("API_KEY")

async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from request header."""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not configured")
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key