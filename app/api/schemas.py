from pydantic import BaseModel, Field
from app.guardrails.trace import DecisionTrace


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    policy_id: str = "default"
    messages: list[Message]
    temperature: float = 0.2
    max_tokens: int = 256


class ChatResponse(BaseModel):
    request_id: str
    model: str
    latency_ms: int
    answer: str
    trace: DecisionTrace


class AnalyzeRequest(BaseModel):
    policy_id: str = "default"
    messages: list[Message]


class AnalyzeResponse(BaseModel):
    policy_id: str
    should_call_llm: bool
    trace: DecisionTrace
    messages: list[Message]  # possibly masked input (e.g., PII masking)

class RewriteRequest(BaseModel):
    text: str
    warn_checks: list[str] = []
    avoid_words: list[str] = []
    temperature: float = 0.2
    max_tokens: int = 120


class RewriteResponse(BaseModel):
    rewritten: str
