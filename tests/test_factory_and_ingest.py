"""
Tests for the LLM factory's honesty contract, ingest versioning, and the
heuristic scorer.  No network access required.
"""

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

import ingest  # noqa: E402
import llm_factory  # noqa: E402
from evaluation.quick_score import quick_evaluate  # noqa: E402


# ------------------------------------------------------------------
# Factory honesty contract
# ------------------------------------------------------------------
# The bug this guards against: the UI used to offer "phi3-mini-finetuned",
# silently serve Groq instead, and keep displaying a fine-tuned badge.  A model
# must never be offered unless it can actually be served.
# ------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "GROQ_API_KEY",
        "HF_TOKEN",
        "PHI3_ENDPOINT_URL",
        "MISTRAL_ENDPOINT_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    # available_models() calls load_dotenv, which would repopulate the vars.
    monkeypatch.setattr(llm_factory, "_load_env", lambda: None)


def test_no_models_offered_without_credentials():
    assert llm_factory.available_models() == []


def test_groq_offered_when_key_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert llm_factory.GROQ_CHOICE in llm_factory.available_models()


def test_finetuned_not_offered_without_endpoint(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    models = llm_factory.available_models()
    assert llm_factory.PHI3_CHOICE not in models
    assert llm_factory.MISTRAL_CHOICE not in models


def test_finetuned_offered_only_with_endpoint_and_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("PHI3_ENDPOINT_URL", "https://example.endpoints.huggingface.cloud")
    models = llm_factory.available_models()
    assert llm_factory.PHI3_CHOICE in models
    assert llm_factory.MISTRAL_CHOICE not in models


def test_endpoint_without_token_is_not_offered(monkeypatch):
    monkeypatch.setenv("PHI3_ENDPOINT_URL", "https://example.endpoints.huggingface.cloud")
    assert llm_factory.PHI3_CHOICE not in llm_factory.available_models()


def test_unconfigured_finetuned_raises_instead_of_falling_back():
    # The critical assertion: no silent substitution of a different model.
    with pytest.raises(EnvironmentError, match="PHI3_ENDPOINT_URL"):
        llm_factory.get_llm(llm_factory.PHI3_CHOICE)


def test_groq_without_key_raises():
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        llm_factory.get_llm(llm_factory.GROQ_CHOICE)


def test_unknown_choice_raises_value_error():
    with pytest.raises(ValueError):
        llm_factory.get_llm("gpt-9-imaginary")


def test_every_choice_has_a_display_label():
    for choice in (
        llm_factory.GROQ_CHOICE,
        llm_factory.PHI3_CHOICE,
        llm_factory.MISTRAL_CHOICE,
    ):
        assert llm_factory.LABELS[choice]


# ------------------------------------------------------------------
# Ingest: document versioning
# ------------------------------------------------------------------
# Guards the stale-document risk: two fee PDFs from different years in one
# corpus, with MMR maximising diversity, can blend years in a single answer.
# ------------------------------------------------------------------
def test_effective_year_from_filename():
    assert ingest.effective_year("fee-structure-2026.pdf") == 2026
    assert ingest.effective_year("Fee-Structure-2025-2.pdf") == 2025


def test_latest_year_wins_for_ranged_filenames():
    assert ingest.effective_year("AC ODD 2025-26_UG-II, III, IV, PG.pdf") == 2025


def test_effective_year_zero_when_absent():
    assert ingest.effective_year("Hostel Facilities.pdf") == 0


def test_effective_year_ignores_non_year_numbers():
    assert ingest.effective_year("scheme16.pdf") == 0


def test_file_digest_detects_identical_content(tmp_path):
    a, b, c = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "c.txt"
    a.write_text("same content")
    b.write_text("same content")
    c.write_text("different")
    assert ingest.file_digest(a) == ingest.file_digest(b)
    assert ingest.file_digest(a) != ingest.file_digest(c)


def test_splitter_respects_chunk_size():
    doc = Document(page_content="Sentence about fees. " * 400, metadata={})
    chunks = ingest.split_documents([doc])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= ingest.CHUNK_SIZE + 50 for c in chunks)


def test_split_preserves_metadata():
    doc = Document(page_content="x " * 900, metadata={"file_name": "f.pdf", "effective_year": 2026})
    for chunk in ingest.split_documents([doc]):
        assert chunk.metadata["effective_year"] == 2026


# ------------------------------------------------------------------
# Heuristic scorer (pure functions, no API)
# ------------------------------------------------------------------
def test_quick_evaluate_returns_three_metrics_in_range():
    docs = [Document(page_content="Hostel fees range from Rs 52,000 per year.", metadata={})]
    scores = quick_evaluate("What are hostel fees?", "Hostel fees start at Rs 52,000.", docs)
    assert set(scores) == {"faithfulness", "answer_relevance", "context_precision"}
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_quick_evaluate_zero_without_context():
    scores = quick_evaluate("q", "some answer that is long enough to score", [])
    assert scores["faithfulness"] == 0.0
    assert scores["context_precision"] == 0.0


def test_grounded_answer_scores_above_ungrounded():
    docs = [Document(page_content="The library is open 24 hours on all 365 days.", metadata={})]
    grounded = quick_evaluate("library hours", "The library is open 24 hours daily.", docs)
    invented = quick_evaluate(
        "library hours", "Elephants migrate across Kenyan savannah plains.", docs
    )
    assert grounded["faithfulness"] > invented["faithfulness"]
