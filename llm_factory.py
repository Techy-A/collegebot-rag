"""
llm_factory.py  --  Multi-LLM Factory for CollegeBot
======================================================
"A designer knows he has achieved perfection not when there is nothing left
 to add, but when there is nothing left to take away."  -- Antoine de Saint-Exupery

This module is the single point of entry for LLM construction.
The application calls get_llm(choice, temperature) and receives
a LangChain-compatible chat model.  All model-specific initialisation
details -- quantization configs, tokenizer hacks, endpoint URLs --
are hidden inside this file.

Supported backends (all free):
    1. groq/llama-3.1-8b-instant  -- Groq Cloud API, ~500 tok/s, no GPU.
    2. phi3-mini-finetuned        -- Fine-tuned Phi-3-mini loaded via HF Hub.
    3. mistral-7b-finetuned       -- Fine-tuned Mistral-7B loaded via HF Hub.

For cloud (Streamlit Community Cloud) deployments, Groq is the only
viable option without a GPU.  The fine-tuned models will automatically
fall back to Groq if no GPU is available and no HF Endpoint URL is set.

Environment variables required (set in .env or Streamlit secrets):
    GROQ_API_KEY       -- from https://console.groq.com (free tier)
    HF_TOKEN           -- from https://huggingface.co/settings/tokens
    PHI3_MODEL_ID      -- e.g. "your-username/collegebot-phi3-mini"
    MISTRAL_MODEL_ID   -- e.g. "your-username/collegebot-mistral-7b"
    PHI3_ENDPOINT_URL  -- optional HF Inference Endpoint for Phi-3
    MISTRAL_ENDPOINT_URL -- optional HF Inference Endpoint for Mistral-7B
"""

import os


def get_llm(choice: str, temperature: float = 0.15):
    """
    Resolve the LLM selection string to a concrete LangChain chat model.

    Parameters
    ----------
    choice      : str   -- One of the three supported keys.
    temperature : float -- Sampling temperature; keep below 0.25 for RAG.

    Returns
    -------
    A LangChain BaseChatModel or BaseLLM compatible with
    ConversationalRetrievalChain.

    "Programs must be written for people to read, and only incidentally
     for machines to execute."  -- Harold Abelson
    """
    dispatch = {
        "groq/llama-3.1-8b-instant": _build_groq,
        "phi3-mini-finetuned"       : _build_phi3,
        "mistral-7b-finetuned"      : _build_mistral,
    }
    if choice not in dispatch:
        raise ValueError(
            f"Unknown LLM choice '{choice}'.  "
            f"Valid options: {list(dispatch.keys())}"
        )
    return dispatch[choice](temperature)


# -----------------------------------------------------------------------
# Backend 1 -- Groq Cloud (llama-3.1-8b-instant)
# -----------------------------------------------------------------------
# Groq's Language Processing Unit (LPU) inference achieves ~500 tokens/s
# on Llama-3.1-8B.  The free tier allows 14,400 requests/day with a
# 6,000 tokens/minute rate limit -- more than sufficient for a college
# chatbot serving hundreds of daily queries.
#
# "Speed is a feature."  -- Unknown, but true for every chatbot user.
# -----------------------------------------------------------------------
def _build_groq(temperature: float):
    from langchain_groq import ChatGroq
    from dotenv import load_dotenv
    from pathlib import Path
    import os

    # Load .env directly inside the factory as a safety net.
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Run in terminal: set GROQ_API_KEY=gsk_yourkey"
        )

    return ChatGroq(
        groq_api_key = api_key,
        model_name   = "llama-3.1-8b-instant",
        temperature  = temperature,
        max_tokens   = 1024,
        streaming    = False,
    )


# -----------------------------------------------------------------------
# Backend 2 -- Fine-tuned Phi-3-mini
# -----------------------------------------------------------------------
# Phi-3-mini-4k-instruct was fine-tuned on the synthetic college Q&A
# dataset using QLoRA (r=16, alpha=32) via Unsloth on a free T4.
# The adapter was merged and pushed to HF Hub.
#
# Loading strategy (in order of preference):
#   a) HF Inference Endpoint URL set -> use endpoint (no local GPU needed)
#   b) GPU present locally            -> load with 4-bit quantization
#   c) No GPU, no endpoint            -> fall back to Groq with a warning
# -----------------------------------------------------------------------
def _build_phi3(temperature: float):
    endpoint_url = os.getenv("PHI3_ENDPOINT_URL", "")
    if endpoint_url:
        return _build_hf_endpoint(endpoint_url, temperature)

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected.")
        model_id = os.getenv("PHI3_MODEL_ID", "your-username/collegebot-phi3-mini")
        return _build_4bit_local(model_id, temperature)
    except Exception as e:
        import warnings
        warnings.warn(
            f"Could not load Phi-3-mini ({e}).  "
            "Falling back to Groq/Llama-3.1-8B.  "
            "Set PHI3_ENDPOINT_URL in .env to use the HF Inference Endpoint."
        )
        return _build_groq(temperature)


# -----------------------------------------------------------------------
# Backend 3 -- Fine-tuned Mistral-7B-Instruct-v0.3
# -----------------------------------------------------------------------
# Mistral-7B was fine-tuned identically to Phi-3-mini but on the larger
# model.  It typically outperforms Phi-3-mini on multi-step reasoning
# but is slower on CPU-only deployments.
#
# The same loading strategy (endpoint -> local 4-bit -> Groq fallback)
# applies here.
# -----------------------------------------------------------------------
def _build_mistral(temperature: float):
    endpoint_url = os.getenv("MISTRAL_ENDPOINT_URL", "")
    if endpoint_url:
        return _build_hf_endpoint(endpoint_url, temperature)

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected.")
        model_id = os.getenv("MISTRAL_MODEL_ID", "your-username/collegebot-mistral-7b")
        return _build_4bit_local(model_id, temperature)
    except Exception as e:
        import warnings
        warnings.warn(
            f"Could not load Mistral-7B ({e}).  "
            "Falling back to Groq/Llama-3.1-8B.  "
            "Set MISTRAL_ENDPOINT_URL in .env to use the HF Inference Endpoint."
        )
        return _build_groq(temperature)


# -----------------------------------------------------------------------
# Shared helper -- HF Inference Endpoint
# -----------------------------------------------------------------------
# HF Inference Endpoints offer a free tier with CPU-only instances.
# Response latency is higher than Groq but the model is your own
# fine-tuned artifact, which is the point.
# -----------------------------------------------------------------------
def _build_hf_endpoint(endpoint_url: str, temperature: float):
    from langchain_huggingface import HuggingFaceEndpoint

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN is not set.  "
            "Create a read token at https://huggingface.co/settings/tokens"
        )
    return HuggingFaceEndpoint(
        endpoint_url          = endpoint_url,
        huggingfacehub_api_token = hf_token,
        task                  = "text-generation",
        temperature           = max(temperature, 0.01),   # HF rejects exactly 0
        max_new_tokens        = 1024,
        repetition_penalty    = 1.1,
        return_full_text      = False,
    )


# -----------------------------------------------------------------------
# Shared helper -- 4-bit local loading (BitsAndBytes + PEFT)
# -----------------------------------------------------------------------
# NF4 quantization halves VRAM usage relative to fp16, making 7B models
# fit comfortably in the 15 GB VRAM of a free Colab T4.
# double_quant=True adds a second quantization pass on the scale factors,
# saving another ~0.4 bits/param with negligible quality loss.
#
# This is the same quantization configuration used during QLoRA training,
# which is not a coincidence -- consistency between training-time and
# inference-time quantization prevents quantization mismatch artifacts.
# -----------------------------------------------------------------------
def _build_4bit_local(model_id: str, temperature: float):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        pipeline,
    )
    from langchain_community.llms import HuggingFacePipeline

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.float16,
        bnb_4bit_use_double_quant = True,
    )

    hf_token  = os.getenv("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code = True,
        token             = hf_token,
    )

    # Ensure a pad token exists.  Many instruction-tuned models omit it.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
        torch_dtype         = torch.float16,
        token               = hf_token,
    )
    model.eval()

    pipe = pipeline(
        "text-generation",
        model          = model,
        tokenizer      = tokenizer,
        max_new_tokens = 1024,
        temperature    = max(temperature, 0.01),
        do_sample      = temperature > 0.01,
        repetition_penalty = 1.1,
        return_full_text   = False,
    )
    return HuggingFacePipeline(pipeline=pipe)
