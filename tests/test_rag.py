"""
Tests for retrieval, prompting and citation formatting.

These run without network access or an API key: the LLM is a stub and the
vector store is built in-memory from a fixture.
"""

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag  # noqa: E402
from prompts import REFUSAL_TEXT, is_refusal  # noqa: E402


# ------------------------------------------------------------------
# Refusal detection
# ------------------------------------------------------------------
def test_refusal_text_is_detected():
    assert is_refusal(REFUSAL_TEXT)


def test_refusal_detection_is_case_insensitive():
    assert is_refusal("I DO NOT HAVE THAT INFORMATION in my knowledge base.")


def test_normal_answer_is_not_a_refusal():
    assert not is_refusal("The annual B.Tech tuition fee is Rs 17,24,000.")


def test_answer_mentioning_information_is_not_a_refusal():
    assert not is_refusal("Here is the information you asked for about hostel fees.")


# ------------------------------------------------------------------
# Follow-up aware retrieval query
# ------------------------------------------------------------------
def test_first_question_query_is_unchanged():
    assert rag.build_retrieval_query("What is the fee?", []) == "What is the fee?"


def test_followup_query_includes_previous_question():
    history = [
        {"role": "user", "content": "What is the B.Tech fee?"},
        {"role": "assistant", "content": "Rs 17,24,000."},
    ]
    query = rag.build_retrieval_query("And for M.Tech?", history)
    assert "B.Tech fee" in query
    assert "M.Tech" in query


def test_followup_uses_only_the_latest_prior_question():
    history = [
        {"role": "user", "content": "first question about hostels"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "second question about fees"},
        {"role": "assistant", "content": "..."},
    ]
    query = rag.build_retrieval_query("what about mess?", history)
    assert "second question about fees" in query
    assert "first question about hostels" not in query


# ------------------------------------------------------------------
# Context and citation formatting
# ------------------------------------------------------------------
def _docs():
    return [
        Document(
            page_content="Hostel fees range from Rs 52,000 to Rs 1,37,000 per year.",
            metadata={"file_name": "thapar_fees.pdf", "page": 3, "effective_year": 2026},
        ),
        Document(
            page_content="The Central Library is open 24 hours.",
            metadata={"file_name": "campus.txt"},
        ),
    ]


def test_context_is_numbered_for_citation():
    ctx = rag.format_context(_docs())
    assert "[1]" in ctx and "[2]" in ctx


def test_context_includes_one_indexed_page_number():
    # PyPDFLoader pages are 0-indexed; users expect 1-indexed.
    assert "page 4" in rag.format_context(_docs())


def test_source_cards_carry_page_and_year():
    cards = rag.source_cards(_docs())
    assert cards[0]["n"] == 1
    assert cards[0]["page"] == 4
    assert cards[0]["year"] == 2026
    assert cards[0]["snippet"].startswith("Hostel fees")


def test_source_card_page_is_none_when_absent():
    assert rag.source_cards(_docs())[1]["page"] is None


def test_history_formatting_labels_speakers():
    text = rag.format_history(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert "Student: hi" in text
    assert "CollegeBot: hello" in text


def test_empty_history_is_explicit():
    assert rag.format_history([]) == "(no previous turns)"


# ------------------------------------------------------------------
# Prompt assembly
# ------------------------------------------------------------------
def test_prompt_contains_question_context_and_refusal_rule():
    prompt = rag.build_prompt("What is the fee?", _docs(), [])
    assert "What is the fee?" in prompt
    assert "52,000" in prompt
    assert "do not have that information" in prompt.lower()


def test_prompt_has_no_unfilled_placeholders():
    prompt = rag.build_prompt("q", _docs(), [])
    assert "{context}" not in prompt
    assert "{question}" not in prompt
    assert "{chat_history}" not in prompt


# ------------------------------------------------------------------
# Streaming and rate-limit handling
# ------------------------------------------------------------------
class _Chunk:
    def __init__(self, content):
        self.content = content


class _StubLLM:
    """Yields chunks; optionally fails the first N attempts with a 429."""

    def __init__(self, chunks, fail_times=0, error="Error code: 429 rate limit reached"):
        self.chunks = chunks
        self.fail_times = fail_times
        self.error = error
        self.attempts = 0

    def stream(self, _prompt):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise RuntimeError(self.error)
        for c in self.chunks:
            yield _Chunk(c)

    def invoke(self, _prompt):
        return _Chunk("".join(self.chunks))


def test_stream_answer_yields_all_chunks():
    llm = _StubLLM(["Hostel ", "fees ", "vary."])
    assert "".join(rag.stream_answer(llm, "p")) == "Hostel fees vary."


def test_stream_answer_retries_rate_limit_then_succeeds():
    llm = _StubLLM(["ok"], fail_times=1)
    assert "".join(rag.stream_answer(llm, "p", backoff=0.01)) == "ok"
    assert llm.attempts == 2


def test_stream_answer_raises_ratelimited_after_retries():
    llm = _StubLLM(["never"], fail_times=99)
    with pytest.raises(rag.RateLimited):
        list(rag.stream_answer(llm, "p", retries=1, backoff=0.01))


def test_non_rate_limit_error_is_not_retried():
    llm = _StubLLM(["x"], fail_times=99, error="ValueError: bad prompt")
    with pytest.raises(RuntimeError):
        list(rag.stream_answer(llm, "p", retries=2, backoff=0.01))
    assert llm.attempts == 1


def test_rate_limit_detection():
    assert rag._is_rate_limit(RuntimeError("Error code: 429"))
    assert rag._is_rate_limit(RuntimeError("Rate limit reached for model"))
    assert not rag._is_rate_limit(RuntimeError("connection reset"))


# ------------------------------------------------------------------
# Hybrid retrieval: Reciprocal Rank Fusion
# ------------------------------------------------------------------
def _d(text):
    return Document(page_content=text, metadata={})


def test_rrf_promotes_documents_found_by_both_retrievers():
    both = _d("appears in both lists")
    dense_only = _d("dense only")
    sparse_only = _d("sparse only")
    fused = rag._rrf([[dense_only, both], [sparse_only, both]])
    assert fused[0].page_content == "appears in both lists"


def test_rrf_deduplicates():
    a, b = _d("alpha"), _d("beta")
    fused = rag._rrf([[a, b], [a, b]])
    assert len(fused) == 2


def test_rrf_respects_rank_order_within_one_list():
    first, second = _d("first"), _d("second")
    fused = rag._rrf([[first, second]])
    assert [d.page_content for d in fused] == ["first", "second"]


def test_rrf_handles_empty_lists():
    assert rag._rrf([[], []]) == []


class _StubStore:
    """Minimal stand-in for a FAISS store."""

    def __init__(self):
        self.similarity_calls = 0
        self.mmr_calls = 0

    def similarity_search(self, _q, k=4):
        self.similarity_calls += 1
        return [_d(f"dense {i}") for i in range(k)]

    def max_marginal_relevance_search(self, _q, k=4, fetch_k=20, lambda_mult=0.5):
        self.mmr_calls += 1
        return [_d(f"mmr {i}") for i in range(k)]


class _StubBM25:
    def get_relevant_documents(self, _q):
        return [_d("sparse hit"), _d("dense 0")]


def test_retrieve_without_bm25_falls_back_to_mmr():
    store = _StubStore()
    docs = rag.retrieve(store, "q", k=3)
    assert store.mmr_calls == 1
    assert len(docs) == 3


def test_retrieve_with_bm25_fuses_both_sources():
    corpus = rag.Corpus(_StubStore(), _StubBM25())
    docs = rag.retrieve(corpus, "q", k=4)
    texts = [d.page_content for d in docs]
    # "dense 0" is returned by both retrievers, so RRF should rank it first.
    assert texts[0] == "dense 0"
    assert "sparse hit" in texts


def test_retrieve_respects_k():
    corpus = rag.Corpus(_StubStore(), _StubBM25())
    assert len(rag.retrieve(corpus, "q", k=2)) == 2


def test_retrieve_accepts_bare_store_for_backwards_compatibility():
    assert rag.retrieve(_StubStore(), "q", k=2)


def test_corpus_ntotal_delegates_to_index():
    class _S:
        class index:
            ntotal = 2185

    assert rag.Corpus(_S()).ntotal == 2185


def test_context_exposes_effective_year_to_the_model():
    # The recency rule in the prompt is only actionable if the year is in context.
    docs = [
        Document(
            page_content="NIRF rank 29 in Engineering.",
            metadata={"file_name": "overview.txt", "effective_year": 2025},
        )
    ]
    assert "year 2025" in rag.format_context(docs)


def test_context_omits_year_when_unknown():
    docs = [Document(page_content="x", metadata={"file_name": "a.txt"})]
    assert "year" not in rag.format_context(docs)


def test_prompt_states_exact_figure_and_recency_rules():
    prompt = rag.build_prompt("q", _docs(), [])
    assert "EXACTLY" in prompt
    assert "most recent year" in prompt


# ------------------------------------------------------------------
# Rate-limit wait handling
# ------------------------------------------------------------------
# The binding limit on the free tier is tokens-per-minute, whose window refills
# over as much as 60s.  A short fixed backoff surfaced an error to the user while
# the budget was still refilling, so the provider's own hint is honoured.
# ------------------------------------------------------------------
def test_retry_after_parsed_from_groq_message():
    err = RuntimeError(
        "Error code: 429 - Rate limit reached for model `llama-3.1-8b-instant`. "
        "Limit 6000, Used 5980. Please try again in 21.9s."
    )
    assert rag.retry_after_seconds(err) == pytest.approx(21.9)


def test_retry_after_parsed_from_header():
    class _Resp:
        headers = {"retry-after": "17"}

    err = RuntimeError("429 too many requests")
    err.response = _Resp()
    assert rag.retry_after_seconds(err) == pytest.approx(17.0)


def test_retry_after_none_when_absent():
    assert rag.retry_after_seconds(RuntimeError("429 too many requests")) is None


def test_retry_honours_provider_advice_over_backoff(monkeypatch):
    waits = []
    monkeypatch.setattr(rag.time, "sleep", lambda s: waits.append(s))
    llm = _StubLLM(["ok"], fail_times=1, error="429 rate limit. Please try again in 12.5s")
    assert "".join(rag.stream_answer(llm, "p")) == "ok"
    assert waits == [pytest.approx(12.5)]


def test_retry_wait_is_capped(monkeypatch):
    waits = []
    monkeypatch.setattr(rag.time, "sleep", lambda s: waits.append(s))
    llm = _StubLLM(["ok"], fail_times=1, error="429 rate limit. Please try again in 900s")
    list(rag.stream_answer(llm, "p", max_wait=65.0))
    assert waits == [65.0]


def test_retry_falls_back_to_exponential_backoff(monkeypatch):
    waits = []
    monkeypatch.setattr(rag.time, "sleep", lambda s: waits.append(s))
    llm = _StubLLM(["ok"], fail_times=2, error="429 too many requests")
    assert "".join(rag.stream_answer(llm, "p", backoff=2.0)) == "ok"
    assert waits == [2.0, 4.0]
