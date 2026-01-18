from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


WarnBlock = Literal["block", "warn"]
MaskBlock = Literal["mask", "block"]


def _default_profanity_words() -> list[str]:
    return ["fuck", "shit", "bitch", "asshole"]


def _default_harm_thresholds() -> dict[str, float]:
    # medium sensitivity default
    return {
        "hate": 0.50,
        "insults": 0.50,
        "sexual": 0.50,
        "violence": 0.50,
    }


def _default_harm_enabled() -> dict[str, bool]:
    return {k: True for k in _default_harm_thresholds().keys()}


class DeniedTopic(BaseModel):
    name: str
    definition: str = ""
    keywords: list[str] = Field(default_factory=list)


class Policy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    policy_id: str = "default"

    # ===== Profanity (warn, not block) =====
    enable_profanity_filter: bool = False
    profanity_action: WarnBlock = "warn"
    profanity_check_input: bool = True
    profanity_check_output: bool = True
    profanity_words: list[str] = Field(default_factory=_default_profanity_words)

    # ===== Blocklist (block) =====
    enable_custom_blocklist: bool = False
    blocklist_action: WarnBlock = "block"
    blocklist_check_input: bool = True
    blocklist_check_output: bool = True
    blocked_phrases: list[str] = Field(default_factory=list)

    # ===== PII (Presidio) =====
    enable_pii_filter: bool = False
    pii_action: MaskBlock = "mask"
    pii_check_input: bool = True
    pii_check_output: bool = True
    pii_types: list[str] = Field(default_factory=list)
    pii_min_confidence: float = 0.5
    pii_mask_token: str = "***"
    pii_language: str = "en"

    # ===== Prompt injection =====
    enable_prompt_injection: bool = False
    prompt_injection_action: WarnBlock = "block"
    prompt_injection_threshold: float = 0.8
    prompt_injection_check_input: bool = True

    # ===== Denied topics (block) =====
    enable_denied_topics: bool = False
    denied_topics_action: WarnBlock = "block"
    denied_topics_check_input: bool = True
    denied_topics_check_output: bool = True
    denied_topics: list[DeniedTopic] = Field(default_factory=list)

    # ===== Harm classes (warn unless hard-block) =====
    enable_harm_classes: bool = False
    harm_classes_action: WarnBlock = "warn"
    harm_classes_check_input: bool = True
    harm_classes_check_output: bool = False

    harm_class_thresholds: dict[str, float] = Field(default_factory=_default_harm_thresholds)
    harm_class_enabled: dict[str, bool] = Field(default_factory=_default_harm_enabled)
    harm_hard_block_threshold: float = 0.90

    # ===== Repair loop =====
    enable_repair: bool = True
    repair_max_attempts: int = 2
    repair_style: Literal["refuse_only", "rewrite_then_refuse"] = "rewrite_then_refuse"
    repair_temperature: float = 0.2
    repair_model: str = ""

    # ===== Later grounding =====
    enable_grounding: bool = False
    grounding_threshold: float = 0.8


_POLICY_DIR = Path(__file__).resolve().parent / "policy_store"
_POLICY_DIR.mkdir(parents=True, exist_ok=True)


def policy_path(policy_id: str) -> Path:
    return _POLICY_DIR / f"{policy_id}.json"


def save_policy(policy: Policy) -> Path:
    path = policy_path(policy.policy_id)
    path.write_text(policy.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_policy(policy_id: str) -> Policy:
    path = policy_path(policy_id)
    if not path.exists():
        raise FileNotFoundError(f"Policy not found: {path}")
    return Policy.model_validate_json(path.read_text(encoding="utf-8"))


def ensure_policy(policy_id: str) -> Policy:
    """
    Single source of truth is the JSON file. If it doesn't exist, create one from Policy defaults.
    """
    path = policy_path(policy_id)
    if path.exists():
        return load_policy(policy_id)
    p = Policy(policy_id=policy_id)
    save_policy(p)
    return p
