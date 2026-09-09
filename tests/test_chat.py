from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import httpx
import pytest

from app.main import app, llm_service
from app.services.llm_service import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_success() -> None:
    with patch.object(llm_service, "chat", new=AsyncMock(return_value="RAG means...")):
        response = client.post("/chat", json={"message": "Explain RAG"})

    assert response.status_code == 200
    assert response.json() == {"response": "RAG means..."}


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {"message": "   "},
        {},
        {"text": "Explain RAG"},
    ],
)
def test_chat_validation_error(payload: dict) -> None:
    response = client.post("/chat", json=payload)
    assert response.status_code == 422


def test_chat_invalid_json() -> None:
    response = client.post(
        "/chat",
        content="not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_chat_missing_api_key() -> None:
    with patch.object(
        llm_service,
        "chat",
        new=AsyncMock(side_effect=LLMAuthError("LLM API key is missing")),
    ):
        response = client.post("/chat", json={"message": "Explain RAG"})

    assert response.status_code == 401
    assert response.json() == {"detail": "LLM API key is missing"}


def test_chat_llm_failure() -> None:
    with patch.object(
        llm_service,
        "chat",
        new=AsyncMock(side_effect=LLMError("LLM service failed")),
    ):
        response = client.post("/chat", json={"message": "Explain RAG"})

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM service failed"}


def test_chat_llm_rate_limit() -> None:
    with patch.object(
        llm_service,
        "chat",
        new=AsyncMock(side_effect=LLMRateLimitError("You exceeded your current quota")),
    ):
        response = client.post("/chat", json={"message": "Explain RAG"})

    assert response.status_code == 429
    assert response.json() == {"detail": "You exceeded your current quota"}


def test_chat_llm_timeout() -> None:
    with patch.object(
        llm_service,
        "chat",
        new=AsyncMock(side_effect=LLMTimeoutError("LLM request timed out")),
    ):
        response = client.post("/chat", json={"message": "Explain RAG"})

    assert response.status_code == 504
    assert response.json() == {"detail": "LLM request timed out"}


def test_llm_service_maps_timeout() -> None:
    from app.config import Settings
    from app.services.llm_service import LLMService

    service = LLMService(Settings(llm_api_key="test-key", llm_timeout=1))

    with patch("app.services.llm_service.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            side_effect=httpx.TimeoutException("timed out")
        )

        with pytest.raises(LLMTimeoutError, match="timed out"):
            import asyncio

            asyncio.run(service.chat("Explain RAG"))
