import os
import requests
import streamlit as st

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:9000")
DEFAULT_POLICY = os.getenv("POLICY_ID", "interactive")


# -----------------------------
# Gateway helpers
# -----------------------------
def gateway_health() -> bool:
    try:
        r = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def get_policy(policy_id: str) -> dict:
    r = requests.get(f"{GATEWAY_URL}/policy/{policy_id}", timeout=15)
    r.raise_for_status()
    return r.json()


def set_policy(policy_id: str, policy: dict) -> None:
    r = requests.post(f"{GATEWAY_URL}/policy/{policy_id}", json=policy, timeout=20)
    r.raise_for_status()


def analyze(policy_id: str, messages: list[dict]) -> dict:
    payload = {"policy_id": policy_id, "messages": messages}
    r = requests.post(f"{GATEWAY_URL}/analyze", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def rewrite_prompt(text: str, warn_checks: list[str], avoid_words: list[str]) -> str:
    payload = {
        "text": text,
        "warn_checks": warn_checks,
        "avoid_words": avoid_words,
        "temperature": 0.2,
        "max_tokens": 120,
    }
    r = requests.post(f"{GATEWAY_URL}/rewrite", json=payload, timeout=120)
    r.raise_for_status()
    return (r.json().get("rewritten") or "").strip()


def chat(policy_id: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
    payload = {
        "policy_id": policy_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"{GATEWAY_URL}/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


# -----------------------------
# Utility
# -----------------------------
def split_multi(s: str) -> list[str]:
    if not s.strip():
        return []
    raw = s.replace(";", ",").split(",")
    return [x.strip() for x in raw if x.strip()]


def parse_denied_topics(text: str) -> list[dict]:
    """
    Each line:
      name | kw1,kw2 | optional definition
    """
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        name = parts[0]
        keywords = split_multi(parts[1])
        definition = parts[2] if len(parts) >= 3 else ""
        out.append({"name": name, "keywords": keywords, "definition": definition})
    return out


def merge_denied_topics(existing: list[dict], additions: list[dict]) -> list[dict]:
    by_name = {t.get("name", "").lower(): t for t in (existing or []) if t.get("name")}
    for a in additions:
        key = a.get("name", "").lower()
        if not key:
            continue
        if key in by_name:
            t = by_name[key]
            ex = {k.lower() for k in (t.get("keywords") or [])}
            for kw in a.get("keywords") or []:
                if kw.lower() not in ex:
                    t.setdefault("keywords", []).append(kw)
        else:
            by_name[key] = {"name": a["name"], "keywords": a.get("keywords") or [], "definition": a.get("definition", "")}
    return list(by_name.values())


def summarize_trace(trace: dict) -> tuple[str, list[str], list[str], list[str]]:
    action = trace.get("action", "allow")
    checks = trace.get("checks_ran") or []
    warns = sorted({c.get("name", "unknown") for c in checks if c.get("triggered") and c.get("action") == "warn"})
    blocks = sorted({c.get("name", "unknown") for c in checks if c.get("triggered") and c.get("action") == "block"})
    warn_check_names = [c.get("name", "") for c in checks if c.get("triggered") and c.get("action") == "warn"]
    return action, warns, blocks, warn_check_names


def extract_avoid_words_from_warns(trace: dict) -> list[str]:
    checks = trace.get("checks_ran") or []
    avoid = []
    for c in checks:
        if c.get("triggered") and c.get("action") == "warn" and str(c.get("name", "")).startswith("profanity_"):
            det = c.get("details") or {}
            m = det.get("matches")
            if isinstance(m, list):
                avoid.extend([str(x) for x in m])
    return sorted(set([x for x in avoid if x]))


def looks_like_model_refusal(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return True
    patterns = [
        "i can't help with that",
        "i can't assist with that",
        "i cannot help with that",
        "i cannot assist with that",
        "i'm not able to help",
        "i’m not able to help",
        "i can't comply",
        "i cannot comply",
        "i can't provide that",
        "i cannot provide that",
    ]
    return any(p in a for p in patterns)


def harm_level_to_threshold(level: str) -> float:
    return {"low": 0.80, "medium": 0.60, "high": 0.40}[level]


# -----------------------------
# Session state
# -----------------------------
def clear_pending_and_result():
    st.session_state.pending = None
    st.session_state.result = None
    st.session_state.pop("rewrite_override", None)


def reset_memory():
    st.session_state.last_assistant = None


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Guardrails Gateway Demo", layout="wide")
st.title("Guardrails Gateway Demo")

ok = gateway_health()

if "last_assistant" not in st.session_state:
    st.session_state.last_assistant = None
if "policy_cache" not in st.session_state:
    st.session_state.policy_cache = None
if "pending" not in st.session_state:
    st.session_state.pending = None
if "result" not in st.session_state:
    st.session_state.result = None
if "main_prompt" not in st.session_state:
    st.session_state.main_prompt = ""

# If user edits the main prompt while a rewrite is pending, drop the pending panel
if st.session_state.pending is not None:
    if st.session_state.main_prompt.strip() != (st.session_state.pending.get("original_prompt") or "").strip():
        st.session_state.pending = None
        st.session_state.pop("rewrite_override", None)


with st.sidebar:
    st.write("Gateway:", GATEWAY_URL)
    st.write("Status:", "✅ OK" if ok else "❌ Not reachable")

    st.header("Load an existing policy")
    policy_id = st.text_input("enter existing policy_id", value=DEFAULT_POLICY)

    # colA, colB = st.columns(2)
    # if colA.button("Load policy", disabled=not ok):
        # st.session_state.policy_cache = get_policy(policy_id)
        # st.success("Loaded.")
    # if colB.button("Reset memory"):
    #     reset_memory()
    #     st.success("Memory reset.")
    load_policy = st.button("Load policy", disabled=not ok)
    if load_policy:
        st.session_state.policy_cache = get_policy(policy_id)
        st.success("Loaded.")



    if st.session_state.policy_cache is None and ok:
        try:
            st.session_state.policy_cache = get_policy(policy_id)
        except Exception:
            st.session_state.policy_cache = None

    policy = st.session_state.policy_cache or {}

    st.divider()
    st.subheader("Quick policy edits")

    add_profanities = st.text_input("Add profanity words (comma-separated)", value="", placeholder="e.g., damn,crap")
    add_blocked = st.text_input("Add blocked words/phrases (comma-separated)", value="", placeholder="e.g., internalcode,topsecret")
    add_denied = st.text_area(
        "Add denied topics (one per line: name | kw1,kw2 | definition)",
        value="",
        height=90,
        placeholder="politics | election,vote | political content",
    )

    enable_pii = st.checkbox("Enable PII masking", value=bool(policy.get("enable_pii_filter", False)))
    enable_injection = st.checkbox("Enable prompt injection check", value=bool(policy.get("enable_prompt_injection", False)))

    harm_level = st.select_slider(
        "Harm sensitivity",
        options=["low", "medium", "high"],
        value="medium",
        help="low = flag only very strong signals (0.80); high = flag more often (0.40).",
    )

    apply_policy_btn = st.button("Apply policy changes to server", disabled=not ok)

    if apply_policy_btn and ok:
        updated = dict(policy)

        # Always-on consistent defaults
        updated["enable_profanity_filter"] = True
        updated["profanity_action"] = "warn"
        updated["enable_custom_blocklist"] = True
        updated["blocklist_action"] = "block"
        updated["enable_harm_classes"] = True
        updated["harm_classes_action"] = "warn"
        updated["harm_hard_block_threshold"] = 0.90
        updated["enable_repair"] = True

        # Fixed thresholds
        updated["enable_prompt_injection"] = bool(enable_injection)
        updated["prompt_injection_threshold"] = 0.8
        updated["prompt_injection_check_input"] = True
        updated["prompt_injection_action"] = "block"

        updated["enable_pii_filter"] = bool(enable_pii)
        updated["pii_action"] = "mask"
        updated["pii_min_confidence"] = 0.6
        if not updated.get("pii_types"):
            updated["pii_types"] = ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON"]

        # Merge additions
        if add_profanities.strip():
            updated.setdefault("profanity_words", [])
            ex = {w.lower() for w in updated["profanity_words"]}
            for w in split_multi(add_profanities):
                if w.lower() not in ex:
                    updated["profanity_words"].append(w)

        if add_blocked.strip():
            updated.setdefault("blocked_phrases", [])
            ex = {p.lower() for p in updated["blocked_phrases"]}
            for p in split_multi(add_blocked):
                if p.lower() not in ex:
                    updated["blocked_phrases"].append(p)

        denied_add = parse_denied_topics(add_denied)
        if denied_add:
            merged = merge_denied_topics(updated.get("denied_topics") or [], denied_add)
            updated["denied_topics"] = merged
            updated["enable_denied_topics"] = True
            updated["denied_topics_action"] = "block"

        # Harm thresholds: one slider applied to all classes
        thr = harm_level_to_threshold(harm_level)
        updated.setdefault("harm_class_thresholds", {})
        updated.setdefault("harm_class_enabled", {})
        for k in ["hate", "insults", "sexual", "violence"]:
            updated["harm_class_thresholds"][k] = min(thr, 0.80)
            updated["harm_class_enabled"][k] = True

        set_policy(policy_id, updated)
        st.session_state.policy_cache = get_policy(policy_id)
        st.success("Policy updated on server.")

    st.divider()
    st.subheader("Request settings")
    max_tokens = st.slider("max_tokens", 32, 1024, 256, step=32)
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, step=0.05)
    rewrite_on_warn = st.checkbox("Suggest rewrite on WARN (like CLI)", value=True)
    show_trace = st.checkbox("Show full trace JSON", value=False)


# -----------------------------
# Main chat UI
# -----------------------------
st.header("Chat")

with st.form("main_send_form", clear_on_submit=False):
    prompt = st.text_area(
        "Your prompt",
        key="main_prompt",
        height=120,
        placeholder="Type a prompt to test guardrails...",
    )
    send_btn = st.form_submit_button("Send", disabled=not ok)

if send_btn:
    clear_pending_and_result()

    user_text = (prompt or "").strip()
    if not user_text:
        st.session_state.result = {
            "status_action": "allow",
            "warns": [],
            "blocks": [],
            "answer": "",
            "trace": {},
            "model_refused": False,
            "pii_masked": False,
            "masked_prompt_sent": None,
        }
        st.rerun()

    base_messages = []
    if st.session_state.last_assistant:
        base_messages.append({"role": "assistant", "content": st.session_state.last_assistant})

    # Precheck (also returns masked messages)
    a = analyze(policy_id, base_messages + [{"role": "user", "content": user_text}])
    trace_pre = a.get("trace", {}) or {}
    action_pre, warns_pre, blocks_pre, warn_check_names = summarize_trace(trace_pre)
    analyzed_messages = a.get("messages") or (base_messages + [{"role": "user", "content": user_text}])

    # Derive what user content would be sent (masked if PII)
    analyzed_user_text = ""
    if analyzed_messages and isinstance(analyzed_messages, list):
        analyzed_user_text = (analyzed_messages[-1].get("content") or "").strip()

    pii_masked = (analyzed_user_text != user_text)

    if action_pre == "warn" and rewrite_on_warn:
        avoid_words = extract_avoid_words_from_warns(trace_pre)
        suggestion = rewrite_prompt(user_text, warn_check_names, avoid_words)

        st.session_state.pending = {
            "policy_id": policy_id,
            "base_messages": base_messages,
            "original_prompt": user_text,
            "suggestion": suggestion,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        st.rerun()

    # Call /chat with the ANALYZED (masked) messages so UI matches what is sent
    resp = chat(policy_id, analyzed_messages, max_tokens, temperature)
    answer = resp.get("answer", "")
    trace = resp.get("trace", {}) or {}
    action, warns, blocks, _ = summarize_trace(trace)

    st.session_state.result = {
        "status_action": action,
        "warns": warns,
        "blocks": blocks,
        "answer": answer,
        "trace": trace,
        "model_refused": (action == "allow" and looks_like_model_refusal(answer)),
        # Only show "prompt sent to model" if not blocked
        "pii_masked": (pii_masked and action != "block"),
        "masked_prompt_sent": (analyzed_user_text if (pii_masked and action != "block") else None),
    }

    if action != "block":
        st.session_state.last_assistant = answer

    st.rerun()


# Pending rewrite panel
if st.session_state.pending is not None:
    p = st.session_state.pending
    st.warning("⚠️ WARNING: policy flagged the input. Suggested rewrite available.")
    st.write("**Suggested rewrite:**")
    st.code(p["suggestion"] or "")

    with st.form("rewrite_choice_form", clear_on_submit=False):
        override = st.text_input(
            "Press Enter to accept suggestion, or type your own prompt:",
            key="rewrite_override",
            value="",
        )
        col1, col2 = st.columns(2)
        go = col1.form_submit_button("Continue (send now)")
        cancel = col2.form_submit_button("Cancel")

    if cancel:
        st.session_state.pending = None
        st.session_state.pop("rewrite_override", None)
        st.rerun()

    if go:
        final_user = p["suggestion"] if (override or "").strip() == "" else override.strip()

        # Analyze again (to capture masking + use masked message for /chat)
        a2 = analyze(p["policy_id"], p["base_messages"] + [{"role": "user", "content": final_user}])
        analyzed_messages2 = a2.get("messages") or (p["base_messages"] + [{"role": "user", "content": final_user}])
        analyzed_user_text2 = (analyzed_messages2[-1].get("content") or "").strip()
        pii_masked2 = (analyzed_user_text2 != final_user)

        resp = chat(p["policy_id"], analyzed_messages2, p["max_tokens"], p["temperature"])
        answer = resp.get("answer", "")
        trace = resp.get("trace", {}) or {}
        action, warns, blocks, _ = summarize_trace(trace)

        st.session_state.result = {
            "status_action": action,
            "warns": warns,
            "blocks": blocks,
            "answer": answer,
            "trace": trace,
            "model_refused": (action == "allow" and looks_like_model_refusal(answer)),
            "pii_masked": (pii_masked2 and action != "block"),
            "masked_prompt_sent": (analyzed_user_text2 if (pii_masked2 and action != "block") else None),
        }

        if action != "block":
            st.session_state.last_assistant = answer

        st.session_state.pending = None
        st.session_state.pop("rewrite_override", None)
        st.rerun()


# Show only ONE latest result
if st.session_state.result is not None:
    r = st.session_state.result
    action = r.get("status_action", "allow")
    warns = list(r.get("warns") or [])
    blocks = r.get("blocks") or []
    answer = r.get("answer", "")
    trace = r.get("trace") or {}

    # Add PII masking as a WARN label
    if r.get("pii_masked") and "pii_masked" not in warns:
        warns.append("pii_masked")

    if action == "block":
        st.error(f"⛔ BLOCKED: {', '.join(blocks) if blocks else 'policy'}")
    elif warns:
        st.warning(f"⚠️ WARNING: {', '.join(sorted(set(warns)))}")
    else:
        st.success("✅ POLICY ALLOWED")

    if r.get("model_refused"):
        st.info("Model refused even though policy allowed (model-side safety/alignment).")

    # Show masked prompt that was actually sent
    masked_prompt_sent = r.get("masked_prompt_sent")
    if masked_prompt_sent:
        st.caption("PII was masked. Prompt sent to the model:")
        st.code(masked_prompt_sent)

    st.subheader("Assistant")
    st.write(answer)

    if show_trace:
        st.subheader("Trace (JSON)")
        st.json(trace)
