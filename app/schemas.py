from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Incoming chat message from the frontend."""

    message: str = Field(..., description="User message to send to the LLM")

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message cannot be empty")
        return cleaned


class ChatResponse(BaseModel):
    """Successful reply from the LLM."""

    response: str = Field(..., description="Generated reply from the LLM")


class ErrorResponse(BaseModel):
    """Standard JSON error body."""

    detail: str


class HealthResponse(BaseModel):
    status: str
