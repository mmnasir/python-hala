from __future__ import annotations

import logging

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base error for LLM integration problems."""


class LLMAuthError(LLMError):
    """Missing or rejected API key."""


class LLMTimeoutError(LLMError):
    """The LLM did not respond in time."""


class LLMRateLimitError(LLMError):
    """LLM rate limit or quota exceeded."""


def _provider_error_message(response: httpx.Response, fallback: str) -> str:
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    except ValueError:
        pass
    return fallback


class LLMService:
    """Calls an OpenAI-compatible chat completions API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def chat(self, message: str) -> str:
        if not self.settings.llm_api_key:
            logger.error("LLM API key is missing")
            raise LLMAuthError("LLM API key is missing")

        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": "user", "content": message}],
        }
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        logger.info("Sending chat request to LLM model=%s", self.settings.llm_model)

        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            logger.error("LLM request timed out")
            raise LLMTimeoutError("LLM request timed out") from exc
        except httpx.RequestError as exc:
            logger.error("LLM service is unavailable: %s", exc)
            raise LLMError("LLM service is unavailable") from exc

        if response.status_code in {401, 403}:
            logger.error("LLM rejected the API key (status %s)", response.status_code)
            raise LLMAuthError("LLM API key is invalid")

        if response.status_code == 429:
            message = _provider_error_message(
                response,
                "LLM rate limit or quota exceeded",
            )
            logger.error("LLM rate limited: %s", message)
            raise LLMRateLimitError(message)

        if response.status_code >= 400:
            message = _provider_error_message(response, "LLM service failed")
            logger.error("LLM service failed (status %s): %s", response.status_code, message)
            raise LLMError(message)

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.error("LLM returned an unexpected response")
            raise LLMError("LLM returned an unexpected response") from exc

        if not isinstance(content, str) or not content.strip():
            logger.error("LLM returned an empty reply")
            raise LLMError("LLM returned an unexpected response")

        logger.info("Received LLM reply")
        return content
