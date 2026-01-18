from typing import Literal

from app.guardrails.policies import Policy
from app.guardrails.trace import DecisionTrace, CheckResult, RepairAttempt
from app.guardrails.filters import find_profanity, find_blocked_phrases
from app.guardrails.pii import analyze_pii, mask_pii
from app.guardrails.injection import score_prompt_injection
from app.guardrails.denied_topics import find_denied_topic_matches
from app.guardrails.repair import propose_safe_response

# If you added harm classes in your interactive build:
# (safe if file exists; otherwise remove these two lines + the helper below)
from app.guardrails.harm_classes import score_harm_classes  # noqa: F401

Stage = Literal["input", "output"]
Action = Literal["allow", "warn", "block"]

BLOCKED_ANSWER = "Blocked by guardrail policy."
FALLBACK_REFUSAL = (
    "I can’t help with that request. If you share what you’re trying to achieve, "
    "I can suggest a safer alternative."
)

_TEXTY_KEYS = {"raw", "text", "content", "candidate", "prompt", "answer", "messages"}


def _final_action_from_checks(checks: list[CheckResult]) -> Action:
    if any(c.triggered and c.action == "block" for c in checks):
        return "block"
    if any(c.triggered and c.action == "warn" for c in checks):
        return "warn"
    return "allow"


def _last_user_index_and_text(messages: list[dict]) -> tuple[int | None, str]:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i, messages[i].get("content", "") or ""
    return None, ""


def _prune_details(checks: list[CheckResult]) -> None:
    # Ensure trace never includes any large/duplicated text fields.
    for c in checks:
        if not isinstance(c.details, dict):
            c.details = {}
            continue
        for k in list(c.details.keys()):
            if k in _TEXTY_KEYS:
                c.details.pop(k, None)


# ---------- Step-3 checks ----------
def _run_step3_checks(text: str, policy: Policy, stage: Stage) -> list[CheckResult]:
    checks: list[CheckResult] = []

    if policy.enable_custom_blocklist and (
        (stage == "input" and policy.blocklist_check_input) or
        (stage == "output" and policy.blocklist_check_output)
    ):
        matches = find_blocked_phrases(text, policy.blocked_phrases)
        triggered = len(matches) > 0
        checks.append(
            CheckResult(
                name=f"custom_blocklist_{stage}",
                triggered=triggered,
                action=policy.blocklist_action if triggered else "none",
                details={"stage": stage, "matches": [m.matched for m in matches]},
            )
        )

    if policy.enable_profanity_filter and (
        (stage == "input" and policy.profanity_check_input) or
        (stage == "output" and policy.profanity_check_output)
    ):
        matches = find_profanity(text, words=getattr(policy, "profanity_words", []))
        triggered = len(matches) > 0
        checks.append(
            CheckResult(
                name=f"profanity_{stage}",
                triggered=triggered,
                action=policy.profanity_action if triggered else "none",
                details={"stage": stage, "matches": [m.matched for m in matches]},
            )
        )

    _prune_details(checks)
    return checks


# ---------- Step-4 PII ----------
def _pii_check_and_maybe_mask(text: str, policy: Policy, stage: Stage) -> tuple[str, CheckResult | None]:
    if not policy.enable_pii_filter:
        return text, None

    enabled_for_stage = (stage == "input" and policy.pii_check_input) or (stage == "output" and policy.pii_check_output)
    if not enabled_for_stage:
        return text, None

    if policy.pii_action == "mask":
        masked, ents = mask_pii(
            text,
            entities=policy.pii_types,
            min_confidence=policy.pii_min_confidence,
            mask_token=policy.pii_mask_token,
            language=policy.pii_language,
        )
        triggered = len(ents) > 0
        check = CheckResult(
            name=f"pii_{stage}",
            triggered=triggered,
            action="mask" if triggered else "none",
            details={"stage": stage, "entity_types": sorted({e.entity_type for e in ents}), "count": len(ents)},
        )
        _prune_details([check])
        return masked, check

    ents = analyze_pii(
        text,
        entities=policy.pii_types,
        min_confidence=policy.pii_min_confidence,
        language=policy.pii_language,
    )
    triggered = len(ents) > 0
    check = CheckResult(
        name=f"pii_{stage}",
        triggered=triggered,
        action="block" if triggered else "none",
        details={"stage": stage, "entity_types": sorted({e.entity_type for e in ents}), "count": len(ents)},
    )
    _prune_details([check])
    return text, check


# ---------- Step-5 Injection ----------
def _prompt_injection_check(user_text: str, policy: Policy) -> CheckResult | None:
    if not policy.enable_prompt_injection or not policy.prompt_injection_check_input:
        return None

    s = score_prompt_injection(user_text, threshold=policy.prompt_injection_threshold)
    check = CheckResult(
        name="prompt_injection_input",
        triggered=s.triggered,
        action=policy.prompt_injection_action if s.triggered else "none",
        details={
            "prob_injection": s.prob_injection,
            "prob_benign": s.prob_benign,
            "threshold": policy.prompt_injection_threshold,
        },
    )
    _prune_details([check])
    return check


# ---------- Step-6 Denied topics ----------
def _denied_topics_check(text: str, policy: Policy, stage: Stage) -> CheckResult | None:
    if not policy.enable_denied_topics:
        return None
    if stage == "input" and not policy.denied_topics_check_input:
        return None
    if stage == "output" and not policy.denied_topics_check_output:
        return None

    matches = find_denied_topic_matches(text, policy.denied_topics)
    triggered = len(matches) > 0
    check = CheckResult(
        name=f"denied_topics_{stage}",
        triggered=triggered,
        action=policy.denied_topics_action if triggered else "none",
        details={"stage": stage, "matches": [{"topic": m.topic_name, "keyword": m.matched_keyword} for m in matches]},
    )
    _prune_details([check])
    return check


def _harm_classes_checks(text: str, policy: Policy, stage: Stage) -> list[CheckResult]:
    if not getattr(policy, "enable_harm_classes", False):
        return []
    if stage == "input" and not getattr(policy, "harm_classes_check_input", True):
        return []
    if stage == "output" and not getattr(policy, "harm_classes_check_output", False):
        return []

    scores = score_harm_classes(text).scores  # {"hate":.., "insults":.., ...}
    out: list[CheckResult] = []

    hard_thr = float(getattr(policy, "harm_hard_block_threshold", 0.90))

    for cls, score in scores.items():
        if not policy.harm_class_enabled.get(cls, True):
            continue

        # cap user threshold at 0.80 (per your requirement)
        user_thr = float(policy.harm_class_thresholds.get(cls, 0.50))
        if user_thr > 0.80:
            user_thr = 0.80
        if user_thr < 0.0:
            user_thr = 0.0

        score_f = float(score)

        if score_f >= hard_thr:
            # hard block regardless of user setting
            out.append(
                CheckResult(
                    name=f"harm_{cls}_{stage}",
                    triggered=True,
                    action="block",
                    details={
                        "score": score_f,
                        "user_threshold": user_thr,
                        "hard_block_threshold": hard_thr,
                        "mode": "hard_block",
                    },
                )
            )
        elif score_f >= user_thr:
            # warn only, still allow model response
            out.append(
                CheckResult(
                    name=f"harm_{cls}_{stage}",
                    triggered=True,
                    action="warn",
                    details={
                        "score": score_f,
                        "user_threshold": user_thr,
                        "hard_block_threshold": hard_thr,
                        "mode": "warn",
                    },
                )
            )
        else:
            out.append(
                CheckResult(
                    name=f"harm_{cls}_{stage}",
                    triggered=False,
                    action="none",
                    details={"score": score_f, "user_threshold": user_thr, "hard_block_threshold": hard_thr},
                )
            )

    # If you have a details-pruning helper, keep it:
    _prune_details(out)
    return out



# ---------- Output evaluation (no repair) ----------
def _evaluate_output(answer: str, policy: Policy) -> tuple[str, list[CheckResult], Action]:
    checks: list[CheckResult] = []

    checks.extend(_run_step3_checks(answer, policy, stage="output"))

    dt = _denied_topics_check(answer, policy, stage="output")
    if dt is not None:
        checks.append(dt)

    # optional harm classes on output (default off)
    checks.extend(_harm_classes_checks(answer, policy, stage="output"))

    masked, pii_check = _pii_check_and_maybe_mask(answer, policy, stage="output")
    if pii_check is not None:
        checks.append(pii_check)

    _prune_details(checks)
    return masked, checks, _final_action_from_checks(checks)


# ---------- Main pre/post APIs ----------
def apply_guardrails_pre(messages: list[dict], policy: Policy) -> tuple[DecisionTrace, bool, list[dict]]:
    """
    Checks only the *latest* user message on input-side checks.
    Returns (trace, should_call_llm, possibly_modified_messages).
    """
    trace = DecisionTrace(policy_id=policy.policy_id, action="allow")
    msg_out = [dict(m) for m in messages]

    idx, last_user_text = _last_user_index_and_text(msg_out)

    # Mask/block PII only on the latest user message
    if idx is not None and msg_out[idx].get("role") == "user":
        new_text, pii_check = _pii_check_and_maybe_mask(last_user_text, policy, stage="input")
        msg_out[idx]["content"] = new_text
        last_user_text = new_text
        if pii_check is not None:
            trace.checks_ran.append(pii_check)
            if pii_check.triggered and policy.pii_action == "mask":
                trace.reasons.append("PII masked in input.")

    # Injection only on latest user message
    inj = _prompt_injection_check(last_user_text, policy)
    if inj is not None:
        trace.checks_ran.append(inj)

    # Denied topics only on latest user message
    dt = _denied_topics_check(last_user_text, policy, stage="input")
    if dt is not None:
        trace.checks_ran.append(dt)

    # Harm classes only on latest user message
    trace.checks_ran.extend(_harm_classes_checks(last_user_text, policy, stage="input"))

    # Step-3 deterministic checks only on latest user message
    trace.checks_ran.extend(_run_step3_checks(last_user_text, policy, stage="input"))

    _prune_details(trace.checks_ran)

    trace.action = _final_action_from_checks(trace.checks_ran)
    if trace.action == "block":
        trace.reasons.append("Input blocked by policy.")
        return trace, False, msg_out
    if trace.action == "warn":
        trace.reasons.append("Input triggered a warning policy.")
    return trace, True, msg_out


def respond_when_blocked_pre(messages: list[dict], policy: Policy, trace: DecisionTrace) -> tuple[str, DecisionTrace]:
    if not policy.enable_repair:
        return BLOCKED_ANSWER, trace

    _, user_text = _last_user_index_and_text(messages)
    blocking_checks = [c for c in trace.checks_ran if c.triggered and c.action == "block"]

    for attempt in range(policy.repair_max_attempts):
        try:
            cand, latency_ms = propose_safe_response(
                phase="pre",
                user_text=user_text,
                original_answer=None,
                policy=policy,
                blocking_checks=blocking_checks,
                attempt=attempt,
            )
        except Exception as e:
            trace.repair_attempts.append(
                RepairAttempt(phase="pre", attempt=attempt, candidate_action="block", blocked_by=[], latency_ms=0, note=f"repair_error: {e}")
            )
            continue

        masked, out_checks, out_action = _evaluate_output(cand or "", policy)
        trace.repair_attempts.append(
            RepairAttempt(
                phase="pre",
                attempt=attempt,
                candidate_action=out_action,
                blocked_by=[c.name for c in out_checks if c.triggered and c.action == "block"],
                latency_ms=latency_ms,
            )
        )
        if out_action != "block" and masked.strip():
            trace.reasons.append("Returned a refusal due to blocked request.")
            trace.checks_ran.extend(out_checks)
            trace.action = "block"
            return masked, trace

    masked, out_checks, out_action = _evaluate_output(FALLBACK_REFUSAL, policy)
    trace.checks_ran.extend(out_checks)
    trace.reasons.append("Returned fallback refusal due to blocked request.")
    trace.action = "block" if out_action == "block" else "block"
    return masked if out_action != "block" else BLOCKED_ANSWER, trace


def apply_guardrails_post(answer: str, policy: Policy, trace: DecisionTrace, user_messages: list[dict]) -> tuple[str, DecisionTrace]:
    masked, out_checks, out_action = _evaluate_output(answer or "", policy)

    if out_action != "block":
        trace.checks_ran.extend(out_checks)
        trace.action = out_action
        return masked, trace

    if not policy.enable_repair:
        trace.checks_ran.extend(out_checks)
        trace.action = "block"
        trace.reasons.append("Output blocked by policy.")
        return BLOCKED_ANSWER, trace

    _, user_text = _last_user_index_and_text(user_messages)

    trace.repair_attempts.append(
        RepairAttempt(
            phase="post",
            attempt=0,
            candidate_action=out_action,
            blocked_by=[c.name for c in out_checks if c.triggered and c.action == "block"],
            latency_ms=0,
            note="initial_output_blocked",
        )
    )
    blocking_checks = [c for c in out_checks if c.triggered and c.action == "block"]

    for attempt in range(policy.repair_max_attempts):
        try:
            cand, latency_ms = propose_safe_response(
                phase="post",
                user_text=user_text,
                original_answer=answer or "",
                policy=policy,
                blocking_checks=blocking_checks,
                attempt=attempt,
            )
        except Exception as e:
            trace.repair_attempts.append(
                RepairAttempt(phase="post", attempt=attempt + 1, candidate_action="block", blocked_by=[], latency_ms=0, note=f"repair_error: {e}")
            )
            continue

        cand_masked, cand_checks, cand_action = _evaluate_output(cand or "", policy)
        trace.repair_attempts.append(
            RepairAttempt(
                phase="post",
                attempt=attempt + 1,
                candidate_action=cand_action,
                blocked_by=[c.name for c in cand_checks if c.triggered and c.action == "block"],
                latency_ms=latency_ms,
            )
        )

        if cand_action != "block" and cand_masked.strip():
            trace.checks_ran.extend(cand_checks)
            trace.action = cand_action
            trace.reasons.append("Output was repaired to satisfy policy.")
            return cand_masked, trace

    final_masked, final_checks, final_action = _evaluate_output(FALLBACK_REFUSAL, policy)
    trace.checks_ran.extend(final_checks)
    trace.action = final_action if final_action != "block" else "block"
    trace.reasons.append("Output was blocked; returned fallback refusal.")
    return final_masked if final_action != "block" else BLOCKED_ANSWER, trace
