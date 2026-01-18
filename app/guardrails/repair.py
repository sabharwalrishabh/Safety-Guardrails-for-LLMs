import os
import time
from typing import Iterable

from openai import OpenAI

from app.guardrails.policies import Policy
from app.guardrails.trace import CheckResult


REPAIR_BASE_URL = os.getenv("REPAIR_BASE_URL", os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"))
REPAIR_API_KEY = os.getenv("REPAIR_API_KEY", os.getenv("LLM_API_KEY", "EMPTY"))

_client = OpenAI(api_key=REPAIR_API_KEY, base_url=REPAIR_BASE_URL)
_cached_model: str = ""


def _get_model_id(policy: Policy) -> str:
    global _cached_model
    if policy.repair_model:
        return policy.repair_model
    if _cached_model:
        return _cached_model
    models = _client.models.list()
    if not models.data:
        raise RuntimeError("No models returned by repair LLM server at " + REPAIR_BASE_URL)
    _cached_model = models.data[0].id
    return _cached_model


def _collect_forbidden(policy: Policy) -> list[str]:
    forb = set()

    # profanity words (if profanity filter is enabled)
    for w in getattr(policy, "profanity_words", []) or []:
        if w and w.strip():
            forb.add(w.strip())

    # blocked phrases
    for p in policy.blocked_phrases:
        if p and p.strip():
            forb.add(p.strip())

    # denied topic keywords
    for t in policy.denied_topics:
        for kw in (t.keywords or []):
            if kw and kw.strip():
                forb.add(kw.strip())

    return sorted(forb, key=len, reverse=True)



def _sanitize_for_prompt(text: str, forbidden: Iterable[str]) -> str:
    # simple case-insensitive replace of forbidden substrings to reduce echo risk
    out = text
    for f in forbidden:
        if not f:
            continue
        # avoid importing regex; do a crude casefold search loop
        low = out.lower()
        f_low = f.lower()
        start = 0
        while True:
            idx = low.find(f_low, start)
            if idx == -1:
                break
            out = out[:idx] + "[REDACTED]" + out[idx + len(f):]
            low = out.lower()
            start = idx + len("[REDACTED]")
    return out


def _blocked_by_names(block_checks: list[CheckResult]) -> list[str]:
    names = []
    for c in block_checks:
        if c.triggered and c.action == "block":
            names.append(c.name)
    return sorted(set(names))


def propose_safe_response(
    *,
    phase: str,  # "pre" or "post"
    user_text: str,
    original_answer: str | None,
    policy: Policy,
    blocking_checks: list[CheckResult],
    attempt: int,
) -> tuple[str, int]:
    """
    Returns (candidate_text, latency_ms).
    """
    forbidden = _collect_forbidden(policy)
    user_s = _sanitize_for_prompt(user_text, forbidden)
    ans_s = _sanitize_for_prompt(original_answer or "", forbidden)

    blocked_by = _blocked_by_names(blocking_checks)

    system = (
        "You generate policy-compliant assistant messages.\n"
        "Constraints:\n"
        "1) Do NOT include any forbidden strings exactly as written.\n"
        "2) Do NOT repeat the user prompt verbatim.\n"
        "3) If the request is disallowed, refuse briefly and offer a safe alternative.\n"
        "4) Keep it short (<= 6 sentences).\n"
        "Forbidden strings:\n"
        f"{forbidden}\n"
    )

    # Make the instruction stronger on later attempts
    if attempt >= 1:
        system += "\nOn retries: be more conservative and prefer refusal."

    # If repair_style is refuse_only, never try to comply
    if policy.repair_style == "refuse_only":
        system += "\nAlways refuse; do not provide the requested content."

    user = (
        f"Phase: {phase}\n"
        f"Blocked_by_checks: {blocked_by}\n"
        f"User_request_sanitized:\n{user_s}\n"
    )
    if original_answer is not None:
        user += f"\nOriginal_answer_sanitized:\n{ans_s}\n"

    model = _get_model_id(policy)

    t0 = time.time()
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=policy.repair_temperature,
        max_tokens=200,
    )
    latency_ms = int((time.time() - t0) * 1000)
    txt = (resp.choices[0].message.content or "").strip()
    return txt, latency_ms
