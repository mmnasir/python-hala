import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from app.services.llm_service import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMService,
    LLMTimeoutError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chat Backend",
    description="Small API that connects a chat frontend to an LLM service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_service = LLMService()


@app.exception_handler(LLMAuthError)
async def llm_auth_error_handler(_, exc: LLMAuthError) -> JSONResponse:
    return JSONResponse(status_code=401, content={"detail": str(exc)})


@app.exception_handler(LLMTimeoutError)
async def llm_timeout_error_handler(_, exc: LLMTimeoutError) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": str(exc)})


@app.exception_handler(LLMRateLimitError)
async def llm_rate_limit_error_handler(_, exc: LLMRateLimitError) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def llm_error_handler(_, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        422: {"description": "Invalid or empty request"},
        429: {"model": ErrorResponse, "description": "LLM rate limit or quota exceeded"},
        502: {"model": ErrorResponse, "description": "LLM service failure"},
        504: {"model": ErrorResponse, "description": "LLM timeout"},
    },
    tags=["chat"],
    summary="Send a message to the LLM",
)
async def chat(payload: ChatRequest) -> ChatResponse:
    logger.info("Received /chat request")
    reply = await llm_service.chat(payload.message)
    return ChatResponse(response=reply)
