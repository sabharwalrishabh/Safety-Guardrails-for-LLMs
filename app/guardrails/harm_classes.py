from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


HARM_MODEL_ID = os.getenv("HARM_MODEL_ID", "unitary/toxic-bert")

_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModelForSequenceClassification] = None
_DEVICE: Optional[torch.device] = None

# Requested classes -> model label names
CLASS_TO_LABEL = {
    "hate": "identity_hate",
    "insults": "insult",
    "sexual": "obscene",
    "violence": "threat",
    # "misconduct": "toxic",
}


@dataclass(frozen=True)
class HarmScores:
    # e.g. {"hate":0.12, "insults":0.88, ...}
    scores: dict[str, float]
    # raw model label scores (optional debugging)
    raw: dict[str, float]


def _ensure_loaded():
    global _TOKENIZER, _MODEL, _DEVICE
    if _TOKENIZER is None or _MODEL is None or _DEVICE is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(HARM_MODEL_ID)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(HARM_MODEL_ID)
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _TOKENIZER, _MODEL, _DEVICE


@torch.inference_mode()
def score_harm_classes(text: str) -> HarmScores:
    tok, model, device = _ensure_loaded()

    inputs = tok(text, truncation=True, max_length=512, return_tensors="pt").to(device)
    out = model(**inputs)
    logits = out.logits.squeeze(0)  # [num_labels]

    # multi-label => sigmoid
    probs = torch.sigmoid(logits)

    # label mapping
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not isinstance(id2label, dict):
        # fallback: assume fixed ordering (rare). still provide requested class scores as 0.
        return HarmScores(scores={k: 0.0 for k in CLASS_TO_LABEL.keys()}, raw={})

    raw_scores = {str(id2label[i]): float(probs[i].item()) for i in range(len(probs))}

    class_scores: dict[str, float] = {}
    for cls, label in CLASS_TO_LABEL.items():
        class_scores[cls] = float(raw_scores.get(label, 0.0))

    return HarmScores(scores=class_scores, raw=raw_scores)
