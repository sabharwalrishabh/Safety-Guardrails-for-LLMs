# Guardrails Gateway Demo (Bedrock-like Safety Wrapper)

A minimal **Bedrock-style safety gateway** for LLMs.

This repo wraps an OpenAI-compatible LLM endpoint (typically **vLLM**) with **configurable guardrails** that can **warn**, **block**, or **mask** content on both **input prompts** and **model outputs**. It also provides:

- a **FastAPI** gateway (policy + analyze + chat + rewrite endpoints)
- an **interactive CLI** for policy editing + 1-turn chat testing
- a **Streamlit UI** for policy editing + chat testing

## What guardrails are implemented?

**Deterministic checks**
- **Custom blocklist** (word/phrase matching; includes obfuscation-resistant matching)
- **Profanity list** (user-extendable; default list provided)
- **Denied topics** (named topics with keyword lists)

**Model-based checks**
- **Harm classes** (multi-label toxicity model; default: `unitary/toxic-bert`) mapped to:
  - `hate`, `insults`, `sexual`, `violence`
- **Prompt injection detection** (binary classifier; default: `protectai/deberta-v3-base-prompt-injection-v2`)

**PII detection**
- Microsoft **Presidio** for PII detection.
- Policy can **mask** PII (default) or **block** when PII is detected.

## Agentic behavior (2 loops)

This project includes two “agentic” behaviors:

1) **Rewrite-on-warn (user-in-the-loop)**
   - If the input is **WARN** (not blocked), the CLI/Streamlit can call `/rewrite` to generate a polite/safer rephrase.
   - The user can accept the suggestion or override it.

2) **Repair loop (LLM-in-the-loop)**
   - If the output is **BLOCK** post-check, the gateway can run a short **repair loop** that asks the LLM to produce a policy-compliant alternative.
   - The repair loop retries up to `repair_max_attempts` and then falls back to a refusal.

## Repository layout

```
safety/
  app/
    main.py                    # FastAPI gateway: /chat, /analyze, /rewrite, /policy/*
    api/schemas.py             # Request/response schemas
    guardrails/
      engine.py                # Orchestrates pre + post checks, repair loop
      policies.py              # Policy model + policy_store JSON persistence
      filters.py               # Blocklist/profanity matching (+ obfuscation handling)
      denied_topics.py         # Topic keyword matching
      pii.py                   # Presidio wrapper (analyze/mask)
      injection.py             # Prompt injection model scoring
      harm_classes.py          # Toxicity/harm class model scoring
      repair.py                # LLM-driven repair loop
      trace.py                 # DecisionTrace + CheckResult models
      policy_store/*.json      # Example policies (interactive, strict, etc.)
  scripts/
    run_guardrail_cli.py       # Interactive CLI tester + policy editor
  ui/
    app.py                     # Streamlit UI
```

## Quickstart

### 0) Prerequisites

- Python 3.10+
- A running **OpenAI-compatible LLM endpoint** (recommended: vLLM)

### 1) Create a virtual environment + install deps

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U fastapi uvicorn openai requests streamlit \
  torch transformers \
  presidio-analyzer presidio-anonymizer spacy

# Presidio typically needs a spaCy model
python -m spacy download en_core_web_sm
```


### 2) Start your LLM server (example: vLLM)

Serve your model with an OpenAI-compatible API. Example (adjust to your environment):

```bash
# Example only — pick the right flags for your machine/model
vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --host 127.0.0.1 --port 8000 --api-key "EMPTY"
```

### 3) Start the FastAPI gateway

```bash
export LLM_BASE_URL="http://127.0.0.1:8000/v1"   # vLLM endpoint

# Optional: pin a specific model ID; otherwise the gateway picks the first from /v1/models
# export LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"

uvicorn app.main:app --host 127.0.0.1 --port 9000 --reload
```

Gateway endpoints:
- `GET /health`
- `GET /policy/{policy_id}` and `POST /policy/{policy_id}` (dev-only; no auth)
- `POST /analyze` (pre-check only)
- `POST /chat` (full pipeline)
- `POST /rewrite` (LLM-powered rewrite suggestion)

### 4) Run the interactive CLI

```bash
python scripts/run_guardrail_cli.py
```

What the CLI does:
- Creates/updates `app/guardrails/policy_store/interactive.json`
- Loop:
  - calls `/analyze` for the prompt
  - if **WARN**, it can suggest a rewrite (user accepts/overrides)
  - calls `/chat` and prints the answer + trace summary
- Maintains **1-turn memory** (previous assistant answer) unless you `/reset`.

### 5) Run the Streamlit UI

```bash
export GATEWAY_URL="http://127.0.0.1:9000"
streamlit run ui/app.py
```

**Screenshot placeholder** (add later):

![Streamlit UI screenshot](docs/streamlit_ui.png)


## Policy configuration

Policies are stored as JSON in `app/guardrails/policy_store/`.

Useful example policies included:
- `interactive.json`: used by CLI/UI for quick edits
- `strict.json`: blocks profanity + blocklist
- `injection_block.json`: enables injection detection
- `pii_mask_both.json`: masks PII on input and output
- `output_only_repair.json`: blocks forbidden topics only on output and uses repair
- `preblock_repair.json`: blocks forbidden topics on input and uses refusal-only repair

