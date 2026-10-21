"""
Unit tests for FAISS PDF RAG API.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import app


client = TestClient(app)

# Test API key
TEST_API_KEY = "my_production_secret_key_12345_change_this"

class TestHealth:
    def test_health_check(self):
        """Test /health endpoint without auth."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "index_ready" in data

class TestAuthentication:
    def test_chat_missing_api_key(self):
        """Test /chat without API key returns 422."""
        response = client.post("/chat", json={"question": "test"})
        assert response.status_code == 422
    
    def test_chat_invalid_api_key(self):
        """Test /chat with wrong API key returns 401."""
        headers = {"X-API-Key": "wrong-key"}
        response = client.post(
            "/chat",
            json={"question": "test"},
            headers=headers
        )
        assert response.status_code == 401

class TestChatEndpoint:
    def test_chat_empty_question(self):
        """Test /chat with empty question."""
        headers = {"X-API-Key": TEST_API_KEY}
        response = client.post(
            "/chat",
            json={"question": ""},
            headers=headers
        )
        assert response.status_code == 400
    
    def test_chat_no_index(self):
        """Test /chat when index doesn't exist."""
        headers = {"X-API-Key": TEST_API_KEY}
        response = client.post(
            "/chat",
            json={"question": "what is this?"},
            headers=headers
        )
        # Should fail because no FAISS index built yet
        assert response.status_code == 400

if __name__ == "__main__":
    pytest.main([__file__, "-v"])