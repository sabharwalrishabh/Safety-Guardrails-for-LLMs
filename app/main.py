import os
import time
from fastapi import FastAPI, HTTPException, Body
from openai import OpenAI

from app.api.schemas import ChatRequest, ChatResponse, AnalyzeRequest, AnalyzeResponse, Message, RewriteRequest, RewriteResponse
from app.guardrails.policies import load_policy, ensure_policy, save_policy, Policy
from app.guardrails.engine import (
    apply_guardrails_pre,
    apply_guardrails_post,
    respond_when_blocked_pre,
)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL = os.getenv("LLM_MODEL", "")

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def _dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()


def get_model_id() -> str:
    global LLM_MODEL
    if LLM_MODEL:
        return LLM_MODEL
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No models returned by LLM server at " + LLM_BASE_URL)
    LLM_MODEL = models.data[0].id
    return LLM_MODEL


app = FastAPI()

@app.get("/policy/{policy_id}")
def get_policy(policy_id: str):
    """
    Returns the policy JSON the gateway is using (stored on the server).
    """
    p = ensure_policy(policy_id)
    return p.model_dump()


@app.post("/policy/{policy_id}")
def set_policy(policy_id: str, payload: dict = Body(...)):
    """
    Overwrites the policy JSON on the server.
    Use only in dev / behind SSH tunnel (not public).
    """
    p = Policy.model_validate(payload)
    p.policy_id = policy_id  # path wins
    save_policy(p)
    return {"ok": True, "policy_id": policy_id}



@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        policy = load_policy(req.policy_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid policy_id: {req.policy_id}. Error: {e}")

    msg_dicts = [_dump(m) for m in req.messages]
    trace, should_call_llm, msg_dicts = apply_guardrails_pre(msg_dicts, policy)

    # return possibly masked messages (PII masking on latest user)
    out_msgs = [Message(**m) for m in msg_dicts]
    return AnalyzeResponse(
        policy_id=req.policy_id,
        should_call_llm=should_call_llm,
        trace=trace,
        messages=out_msgs,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        policy = load_policy(req.policy_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid policy_id: {req.policy_id}. Error: {e}")

    msg_dicts = [_dump(m) for m in req.messages]

    # PRE checks (may mask or block before LLM)
    trace, should_call_llm, msg_dicts = apply_guardrails_pre(msg_dicts, policy)

    model = get_model_id()
    if not should_call_llm:
        answer, trace = respond_when_blocked_pre(msg_dicts, policy, trace)
        return ChatResponse(
            request_id="blocked_pre",
            model=model,
            latency_ms=0,
            answer=answer,
            trace=trace,
        )

    # Call LLM
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msg_dicts,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((time.time() - t0) * 1000)
    answer = resp.choices[0].message.content or ""

    # POST checks (may mask/block, then repair if needed)
    final_answer, trace = apply_guardrails_post(answer, policy, trace, user_messages=msg_dicts)

    return ChatResponse(
        request_id=getattr(resp, "id", "local"),
        model=model,
        latency_ms=latency_ms,
        answer=final_answer,
        trace=trace,
    )

@app.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    model = get_model_id()

    system = (
        "Rewrite the user's message to be polite and safe while preserving the original intent.\n"
        "Rules:\n"
        "- Remove profanity and aggressive phrasing.\n"
        "- Do not add new requests or new content.\n"
        "- Keep it as ONE user message.\n"
        "- Output ONLY the rewritten message (no quotes, no explanations).\n"
    )
    if req.avoid_words:
        system += f"\nAvoid these words: {req.avoid_words}\n"

    user = f"User message:\n{req.text}\n\nWarnings seen: {req.warn_checks}\n"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    out = (resp.choices[0].message.content or "").strip()
    out = " ".join(out.split())
    return RewriteResponse(rewritten=out)
