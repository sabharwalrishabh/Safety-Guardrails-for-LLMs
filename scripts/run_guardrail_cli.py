import os
import requests
from openai import OpenAI
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.guardrails.policies import ensure_policy, save_policy, Policy, DeniedTopic

GATEWAY = os.getenv("GATEWAY_URL", "http://127.0.0.1:9000")

# vLLM for rewrite suggestions
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "")

rewrite_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
_cached_rewrite_model = ""

POLICY_ID = "interactive"


def _get_rewrite_model_id() -> str:
    global _cached_rewrite_model
    if REWRITE_MODEL:
        return REWRITE_MODEL
    if _cached_rewrite_model:
        return _cached_rewrite_model
    models = rewrite_client.models.list()
    if not models.data:
        raise RuntimeError("No models returned by LLM server at " + LLM_BASE_URL)
    _cached_rewrite_model = models.data[0].id
    return _cached_rewrite_model


def _split_multi(s: str) -> list[str]:
    if not s.strip():
        return []
    raw = s.replace(";", ",").split(",")
    return [x.strip() for x in raw if x.strip()]


def _yes(prompt: str, default_yes: bool = True) -> bool:
    d = "Y/n" if default_yes else "y/N"
    ans = input(f"{prompt} [{d}]: ").strip().lower()
    if not ans:
        return default_yes
    return ans.startswith("y")


def _pick_level(prompt: str) -> str:
    ans = input(f"{prompt} [low/medium/high] (default medium): ").strip().lower()
    return ans if ans in ("low", "medium", "high") else "medium"


def _level_to_thr(level: str) -> float:
    # low sensitivity => high numeric threshold
    return {"low": 0.80, "medium": 0.60, "high": 0.40}[level]


def _maybe_float(prompt: str, current: float) -> float:
    s = input(f"{prompt} (Enter to keep {current}): ").strip()
    if not s:
        return current
    try:
        return float(s)
    except ValueError:
        print("Invalid number; keeping current.")
        return current


def _configure_denied_topics(policy: Policy):
    print("\nDenied topics: one per line as:")
    print("  name | keyword1,keyword2 | optional definition")
    print("Press Enter on empty line to finish.\n")

    by_name = {t.name.lower(): t for t in policy.denied_topics}

    while True:
        line = input("Denied topic: ").strip()
        if not line:
            break
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            print("  Invalid. Need at least: name | keywords")
            continue

        name = parts[0]
        keywords = _split_multi(parts[1])
        definition = parts[2] if len(parts) >= 3 else ""

        key = name.lower()
        if key in by_name:
            t = by_name[key]
            existing = {k.lower() for k in (t.keywords or [])}
            for kw in keywords:
                if kw.lower() not in existing:
                    t.keywords.append(kw)
        else:
            t = DeniedTopic(name=name, keywords=keywords, definition=definition)
            policy.denied_topics.append(t)
            by_name[key] = t


def write_or_update_policy_interactively() -> Policy:
    # Load existing policy JSON or create it once from defaults.
    policy = ensure_policy(POLICY_ID)

    print(f"=== Policy file used: app/guardrails/policy_store/{POLICY_ID}.json ===")

    policy.enable_profanity_filter = True
    policy.profanity_action = "warn"

    policy.enable_custom_blocklist = True
    policy.blocklist_action = "block"

    policy.enable_harm_classes = True
    policy.harm_classes_action = "warn"

    policy.enable_repair = True

    # Lists: append only if user adds
    add_prof = _split_multi(input("Add profanity words (comma-separated) or blank: "))
    if add_prof:
        existing = {w.lower() for w in policy.profanity_words}
        for w in add_prof:
            if w.lower() not in existing:
                policy.profanity_words.append(w)

    add_block = _split_multi(input("Add blocked words/phrases (comma-separated) or blank: "))
    if add_block:
        existing = {p.lower() for p in policy.blocked_phrases}
        for p in add_block:
            if p.lower() not in existing:
                policy.blocked_phrases.append(p)

    if _yes("Add denied topics now?", default_yes=False):
        policy.enable_denied_topics = True
        policy.denied_topics_action = "block"
        _configure_denied_topics(policy)

    # Harm sensitivity controls (updates the JSON thresholds)
    if _yes("Adjust harm class sensitivity (low/medium/high)?", default_yes=True):
        for cls in ["hate", "insults", "sexual", "violence"]:
            level = _pick_level(f"  Sensitivity for {cls}")
            policy.harm_class_thresholds[cls] = _level_to_thr(level)
            policy.harm_class_enabled[cls] = True

    # Prompt injection threshold (updates the JSON)
    if _yes("Enable prompt injection guardrail?", default_yes=True):
        policy.enable_prompt_injection = True
        policy.prompt_injection_threshold = 0.8


    # PII settings (updates the JSON)
    if _yes("Enable PII guardrail?", default_yes=True):
        policy.enable_pii_filter = True
        policy.pii_min_confidence = 0.6


    save_policy(policy)
    print("\nSaved policy JSON. Future runs will use the same file.\n")
    return policy


def _triggered_checks(trace: dict, action: str | None = None) -> list[dict]:
    checks = trace.get("checks_ran") or []
    out = []
    for c in checks:
        if not c.get("triggered"):
            continue
        if action is not None and c.get("action") != action:
            continue
        out.append(c)
    return out


def _warn_banner(trace: dict):
    warns = _triggered_checks(trace, action="warn")
    if not warns:
        return
    names = sorted({w.get("name", "unknown") for w in warns})
    print(f"\n⚠️  WARNING: {', '.join(names)}")


def _block_banner(trace: dict):
    blocks = _triggered_checks(trace, action="block")
    if not blocks:
        return
    names = sorted({b.get("name", "unknown") for b in blocks})
    print(f"\n⛔ BLOCKED: {', '.join(names)}")


def _print_repair_report(trace: dict):
    reps = trace.get("repair_attempts") or []
    if not reps:
        return
    phase = reps[-1].get("phase", "?")
    attempts = len(reps)
    succeeded = any(a.get("candidate_action") in ("allow", "warn") for a in reps)
    print(f"\n🔁 REPAIR LOOP USED: phase={phase} attempts={attempts} success={succeeded}")


def _extract_warning_hints(trace: dict) -> dict:
    warns = _triggered_checks(trace, action="warn")
    names = sorted({w.get("name", "") for w in warns if w.get("name")})

    prof = []
    for w in warns:
        if w.get("name", "").startswith("profanity_"):
            det = w.get("details") or {}
            m = det.get("matches")
            if isinstance(m, list):
                prof.extend([str(x) for x in m])

    return {"warn_checks": names, "profanity_matches": sorted(set([x for x in prof if x]))}


def suggest_rewrite(user_text: str, warn_hints: dict) -> str:
    model = _get_rewrite_model_id()
    warn_checks = warn_hints.get("warn_checks", [])
    prof = warn_hints.get("profanity_matches", [])

    system = (
        "Rewrite the user's message to be polite and safe while preserving the original intent.\n"
        "Rules:\n"
        "- Remove profanity and aggressive phrasing.\n"
        "- Do not add new requests or new content.\n"
        "- Keep it as ONE user message.\n"
        "- Output ONLY the rewritten message (no quotes, no explanations).\n"
    )
    if prof:
        system += f"\nAvoid these words: {prof}\n"

    user = f"User message:\n{user_text}\n\nWarnings seen: {warn_checks}\n"

    resp = rewrite_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=120,
    )
    out = (resp.choices[0].message.content or "").strip()
    return " ".join(out.split())


def call_analyze(policy_id: str, messages: list[dict]) -> dict:
    payload = {"policy_id": policy_id, "messages": messages}
    r = requests.post(f"{GATEWAY}/analyze", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def call_chat(policy_id: str, messages: list[dict]) -> dict:
    payload = {"policy_id": policy_id, "messages": messages, "max_tokens": 256, "temperature": 0.2}
    r = requests.post(f"{GATEWAY}/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def chat_loop():
    r = requests.get(f"{GATEWAY}/health", timeout=10)
    r.raise_for_status()

    print("\n=== 1-turn memory prompt tester + rewrite-on-warn ===")
    print("Commands: /reset, /exit\n")

    last_assistant = None

    while True:
        user = input("you> ").strip()
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            last_assistant = None
            print("(memory reset)")
            continue

        base_messages = []
        if last_assistant is not None:
            base_messages.append({"role": "assistant", "content": last_assistant})

        # Precheck via /analyze
        analyze_messages = base_messages + [{"role": "user", "content": user}]
        a = call_analyze(POLICY_ID, analyze_messages)
        trace = a.get("trace", {})
        action = trace.get("action", "allow")

        final_user = user
        if action == "warn":
            _warn_banner(trace)
            hints = _extract_warning_hints(trace)
            try:
                suggestion = suggest_rewrite(user, hints)
            except Exception as e:
                suggestion = ""
                print(f"(rewrite suggestion failed: {e})")

            if suggestion:
                print("\nSuggested rewrite:")
                print(f"> {suggestion}")
                override = input("\nPress Enter to send this, or type your own prompt:\n> ").strip()
                final_user = suggestion if override == "" else override

        # Send to /chat
        chat_messages = base_messages + [{"role": "user", "content": final_user}]
        j = call_chat(POLICY_ID, chat_messages)

        answer = j.get("answer", "")
        trace = j.get("trace", {})

        _block_banner(trace)
        _warn_banner(trace)
        print("\nassistant>", answer)

        _print_repair_report(trace)

        if trace.get("action") != "block":
            last_assistant = answer


if __name__ == "__main__":
    write_or_update_policy_interactively()
    chat_loop()
