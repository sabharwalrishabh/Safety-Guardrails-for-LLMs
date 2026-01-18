from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Recommended: use v2 by default (newer). You can override with env var.
DEFAULT_MODEL_ID = os.getenv(
    "INJECTION_MODEL_ID",
    "protectai/deberta-v3-base-prompt-injection-v2",
)

_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModelForSequenceClassification] = None
_DEVICE: Optional[torch.device] = None


@dataclass(frozen=True)
class InjectionScore:
    prob_injection: float
    prob_benign: float
    triggered: bool


def _ensure_loaded() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, torch.device]:
    global _TOKENIZER, _MODEL, _DEVICE
    if _TOKENIZER is None or _MODEL is None or _DEVICE is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_ID)
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _TOKENIZER, _MODEL, _DEVICE


@torch.inference_mode()
def score_prompt_injection(text: str, threshold: float = 0.5) -> InjectionScore:
    """
    ProtectAI prompt-injection models are documented as:
      class 0 = benign
      class 1 = injection
    We treat index 1 as injection explicitly (no guessing).
    """
    tok, model, device = _ensure_loaded()

    inputs = tok(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    out = model(**inputs)
    logits = out.logits.squeeze(0)  # [2]
    probs = torch.softmax(logits, dim=-1)

    prob_benign = float(probs[0].item())
    prob_inj = float(probs[1].item())

    return InjectionScore(
        prob_injection=prob_inj,
        prob_benign=prob_benign,
        triggered=(prob_inj >= threshold),
    )
