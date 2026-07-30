"""
llm_factory.py  --  LLM backend factory for CollegeBot
=======================================================
Single point of entry for LLM construction.  The application calls
available_models() to learn what can actually be served right now, and
get_llm(choice, temperature) to build it.

HONESTY CONTRACT
----------------
This module previously advertised three backends -- Groq, a fine-tuned Phi-3
and a fine-tuned Mistral-7B -- and, when the fine-tuned models could not be
loaded, silently returned Groq while the UI went on displaying a
"Phi-3-mini (fine-tuned)" badge.  The warning went to stderr, where no user
would ever see it.  The UI therefore asserted something false about which model
produced the answer.

The rule now: a backend is only offered if it can actually be served.
available_models() checks configuration and returns only usable choices, so the
UI cannot present a model it will not use.  There is no silent fallback.

Backends
--------
    groq/llama-3.1-8b-instant   Groq Cloud API.  Requires GROQ_API_KEY.
    phi3-mini-finetuned         Requires PHI3_ENDPOINT_URL + HF_TOKEN.
    mistral-7b-finetuned        Requires MISTRAL_ENDPOINT_URL + HF_TOKEN.

The fine-tuned backends are served through Hugging Face Inference Endpoints.
The QLoRA training code that produces those adapters lives in
notebooks/CollegeBot_Colab.py; until an endpoint is deployed and its URL is set,
those options simply do not appear.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

GROQ_CHOICE = f"groq/{GROQ_MODEL}"
PHI3_CHOICE = "phi3-mini-finetuned"
MISTRAL_CHOICE = "mistral-7b-finetuned"

LABELS = {
    GROQ_CHOICE: "Groq / Llama-3.1-8B",
    PHI3_CHOICE: "Phi-3-mini (fine-tuned)",
    MISTRAL_CHOICE: "Mistral-7B (fine-tuned)",
}


def _load_env() -> None:
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)


def available_models() -> List[str]:
    """
    Backends that can actually answer a question right now.

    Groq is listed when GROQ_API_KEY is present.  A fine-tuned model is listed
    only when its Inference Endpoint URL and an HF token are both configured.
    """
    _load_env()
    models = []
    if os.getenv("GROQ_API_KEY", "").strip():
        models.append(GROQ_CHOICE)
    if os.getenv("HF_TOKEN", "").strip():
        if os.getenv("PHI3_ENDPOINT_URL", "").strip():
            models.append(PHI3_CHOICE)
        if os.getenv("MISTRAL_ENDPOINT_URL", "").strip():
            models.append(MISTRAL_CHOICE)
    return models


def get_llm(choice: str, temperature: float = 0.05, streaming: bool = True):
    """
    Resolve a selection string to a LangChain chat model.

    Raises EnvironmentError if the requested backend is not configured -- it
    never quietly substitutes a different model.
    """
    _load_env()

    if choice == GROQ_CHOICE:
        return _build_groq(temperature, streaming)

    if choice == PHI3_CHOICE:
        return _build_hf_endpoint(_require("PHI3_ENDPOINT_URL", choice), temperature)

    if choice == MISTRAL_CHOICE:
        return _build_hf_endpoint(_require("MISTRAL_ENDPOINT_URL", choice), temperature)

    raise ValueError(f"Unknown LLM choice '{choice}'.  Configured options: {available_models()}")


def _require(var: str, choice: str) -> str:
    value = os.getenv(var, "").strip()
    if not value:
        raise OSError(
            f"'{choice}' is not available: {var} is not set.  "
            "Deploy a Hugging Face Inference Endpoint for the fine-tuned model "
            f"and set {var} in .env, or select {GROQ_CHOICE}."
        )
    return value


# -----------------------------------------------------------------------
# Groq Cloud
# -----------------------------------------------------------------------
# Groq's LPU inference runs Llama-3.1-8B at roughly 500 tokens/s.  The free
# tier allows 14,400 requests/day with a 30 requests/minute ceiling, which is
# the binding constraint for this app -- not model speed.
# -----------------------------------------------------------------------
def _build_groq(temperature: float, streaming: bool = True):
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise OSError(
            "GROQ_API_KEY is not set.  Create a free key at "
            "https://console.groq.com/keys and add it to .env as "
            "GROQ_API_KEY=gsk_..."
        )

    return ChatGroq(
        groq_api_key=api_key,
        model_name=GROQ_MODEL,
        temperature=temperature,
        # Output tokens count against the same 6,000 tokens/minute budget as the
        # prompt.  Observed answers run 150-250 tokens, so 700 leaves headroom
        # for long lists without spending the minute's budget on one reply.
        max_tokens=700,
        streaming=streaming,
        timeout=60,
        max_retries=2,
    )


# -----------------------------------------------------------------------
# Hugging Face Inference Endpoint (fine-tuned models)
# -----------------------------------------------------------------------
def _build_hf_endpoint(endpoint_url: str, temperature: float):
    from langchain_huggingface import HuggingFaceEndpoint

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise OSError(
            "HF_TOKEN is not set.  Create a read token at https://huggingface.co/settings/tokens"
        )
    return HuggingFaceEndpoint(
        endpoint_url=endpoint_url,
        huggingfacehub_api_token=hf_token,
        task="text-generation",
        temperature=max(temperature, 0.01),  # HF rejects exactly 0
        max_new_tokens=1024,
        repetition_penalty=1.1,
        return_full_text=False,
    )
