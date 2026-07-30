"""
rag.py  --  Retrieval and answer generation for CollegeBot
==========================================================
This module owns the RAG pipeline.  It is deliberately free of Streamlit UI
code so the evaluation harness can import and exercise exactly the same
retrieval and prompting path the live app uses.

Two design decisions matter here:

1.  ONLY the stateless, expensive objects are cached process-wide (the
    embedding model and the FAISS index).  Conversation memory is NOT cached --
    it belongs to a single user's session.  The previous implementation built
    the memory inside an @st.cache_resource function, which is a process-global
    singleton on Streamlit, so concurrent visitors shared one mutable memory
    buffer and their conversations bled into each other's context window.

2.  No ConversationalRetrievalChain.  That legacy chain fires an extra
    "condense question" LLM call before retrieval on every follow-up turn,
    which doubles request count against a 30 req/min free tier, and it does not
    stream the final answer cleanly.  Retrieve -> prompt -> stream is both
    cheaper and directly streamable.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, Iterator, List, Tuple

from langchain_core.documents import Document

from prompts import QA_TEMPLATE

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_FAISS_PATH = os.getenv("FAISS_PATH", "./faiss_store")

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "30"))
RETRIEVAL_LAMBDA = float(os.getenv("RETRIEVAL_LAMBDA", "0.5"))

# Hybrid retrieval: dense (FAISS) fused with sparse (BM25) by Reciprocal Rank
# Fusion.  On by default because it was measured, not assumed.
#
# Measured fact-recall on the 35 answerable gold questions (does the key figure
# reach the context at all?):
#
#     MMR k=6 fetch_k=30              77.1%   (the previous default)
#     BM25 only, k=8                  80.0%
#     dense(60) + cross-encoder -> 8  88.6%   but ~2s/query and a model load
#     hybrid dense+BM25 RRF -> 8      88.6%   and no added latency
#     hybrid + cross-encoder -> 8     85.7%   reranking on top made it worse
#
# Hybrid wins on cost, ties on accuracy.  Sparse retrieval earns its place here
# because the corpus is full of proper nouns, exact figures and list headings
# ("TOP RECRUITERS", "Rs 17,24,000", "A++") where lexical match beats a 384-dim
# dense embedding.
USE_HYBRID = os.getenv("USE_HYBRID", "1").strip() not in ("0", "false", "")
BM25_K = int(os.getenv("BM25_K", "30"))
RRF_K = 60  # standard RRF damping constant

# Opt-in cross-encoder reranking.  Left in because it is a reasonable lever on a
# different corpus, but off by default: on THIS corpus it did not beat hybrid.
USE_RERANKER = os.getenv("USE_RERANKER", "").strip() not in ("", "0", "false")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

MAX_HISTORY_TURNS = 5

log = logging.getLogger("collegebot")


def configure_logging() -> None:
    """Structured-ish stdout logging.  Streamlit Cloud captures stdout."""
    if log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s collegebot %(message)s"))
    log.addHandler(handler)
    log.setLevel(os.getenv("LOG_LEVEL", "INFO"))


# ------------------------------------------------------------------
# Vector store  (expensive, stateless, safe to share across sessions)
# ------------------------------------------------------------------
def load_vectorstore(faiss_path: str | None = None):
    """
    Load the embedding model and FAISS index.

    Keyed only on the store path -- not on the model choice or temperature --
    so changing a generation parameter never re-reads the index from disk.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    faiss_path = faiss_path or DEFAULT_FAISS_PATH
    if not os.path.isdir(faiss_path):
        raise FileNotFoundError(f"No FAISS store found at '{faiss_path}'.  Run: python ingest.py")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # allow_dangerous_deserialization: this index is built by our own ingest.py.
    store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    log.info("vectorstore loaded path=%s vectors=%d", faiss_path, store.index.ntotal)
    return store


class Corpus:
    """A dense index plus an optional sparse index over the same chunks."""

    def __init__(self, store, bm25=None):
        self.store = store
        self.bm25 = bm25

    @property
    def ntotal(self) -> int:
        return self.store.index.ntotal


def load_corpus(faiss_path: str | None = None) -> Corpus:
    """Load the dense index and, if hybrid retrieval is on, build the BM25 index."""
    store = load_vectorstore(faiss_path)
    bm25 = None
    if USE_HYBRID:
        try:
            from langchain_community.retrievers import BM25Retriever

            chunks = list(store.docstore._dict.values())
            bm25 = BM25Retriever.from_documents(chunks)
            bm25.k = BM25_K
            log.info("bm25 index built chunks=%d", len(chunks))
        except Exception as exc:  # noqa: BLE001 - hybrid is an optimisation
            log.warning("bm25 unavailable, dense-only retrieval: %s", exc)
    return Corpus(store, bm25)


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
        log.info("reranker loaded model=%s", RERANKER_MODEL)
    return _reranker


def _rrf(ranked_lists: List[List[Document]]) -> List[Document]:
    """
    Reciprocal Rank Fusion: score each document by sum(1 / (K + rank)) across the
    ranked lists it appears in.  Rank-based, so it needs no score normalisation
    between two retrievers with incomparable scales (cosine vs BM25).
    """
    scores: Dict[str, float] = {}
    docs: Dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content[:160]
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    return [docs[k] for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
def build_retrieval_query(question: str, history: List[Dict]) -> str:
    """
    Make follow-ups retrievable without spending an extra LLM call.

    "What about for M.Tech?" carries no retrievable content on its own, so the
    previous user question is prepended.  A condense-question LLM call would be
    marginally better but costs a second request per turn against a 30 req/min
    budget.
    """
    prior_user = [m["content"] for m in history if m.get("role") == "user"]
    if not prior_user:
        return question
    return f"{prior_user[-1]} {question}"


def retrieve(corpus, query: str, k: int = RETRIEVAL_K) -> List[Document]:
    """
    Retrieve the k passages to ground the answer on.

    Accepts a Corpus (hybrid-capable) or a bare FAISS store (dense only), so
    callers that only have a vector store keep working.
    """
    store = getattr(corpus, "store", corpus)
    bm25 = getattr(corpus, "bm25", None)
    fetch_k = RETRIEVAL_FETCH_K

    dense = store.similarity_search(query, k=fetch_k)

    if bm25 is not None:
        sparse = bm25.get_relevant_documents(query)
        candidates = _rrf([dense, sparse])
    else:
        # Dense only: MMR, to avoid returning k near-duplicate passages.
        candidates = store.max_marginal_relevance_search(
            query, k=max(k, RETRIEVAL_K), fetch_k=fetch_k, lambda_mult=RETRIEVAL_LAMBDA
        )

    if USE_RERANKER:
        pool = candidates[: max(40, k * 5)]
        scores = _get_reranker().predict([(query, d.page_content) for d in pool])
        candidates = [d for _, d in sorted(zip(scores, pool), key=lambda p: -p[0])]

    return candidates[:k]


def format_context(docs: List[Document]) -> str:
    """
    Number the chunks so the model can cite them and the UI can line citations
    up with the sources panel.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        loc = f"{src}, page {page + 1}" if isinstance(page, int) else str(src)
        # The effective year must reach the model, not just the citation UI.
        # Without it the prompt's "prefer the most recent source" rule is
        # unusable -- the model cannot tell a 2018 self-study report from a 2026
        # fee circular, and will happily quote the stale figure.
        year = doc.metadata.get("effective_year")
        if year:
            loc += f", year {year}"
        parts.append(f"[{i}] ({loc})\n{doc.page_content}")
    return "\n\n".join(parts)


def format_history(history: List[Dict], turns: int = MAX_HISTORY_TURNS) -> str:
    recent = history[-turns * 2 :]
    if not recent:
        return "(no previous turns)"
    return "\n".join(
        f"{'Student' if m.get('role') == 'user' else 'CollegeBot'}: {m.get('content', '')}"
        for m in recent
    )


def source_cards(docs: List[Document]) -> List[Dict]:
    """UI-ready citation records: numbered, named, page-located, with the snippet."""
    cards = []
    for i, doc in enumerate(docs, 1):
        raw = doc.metadata.get("file_name") or doc.metadata.get("source", "unknown")
        name = os.path.basename(str(raw))
        page = doc.metadata.get("page")
        cards.append(
            {
                "n": i,
                "file": name,
                "title": _pretty_title(name),
                "page": page + 1 if isinstance(page, int) else None,
                "year": doc.metadata.get("effective_year"),
                "snippet": doc.page_content.strip(),
            }
        )
    return cards


def _pretty_title(filename: str) -> str:
    """'AC ODD 2025-26_UG-II, III, IV, PG.pdf' -> something a student can read."""
    stem = os.path.splitext(filename)[0]
    stem = stem.replace("_", " ").replace("-", " ")
    stem = " ".join(stem.split())
    return stem.title() if stem.islower() or stem.isupper() else stem


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------
class RateLimited(Exception):
    """Groq (or another provider) refused the request due to rate limiting."""


def _is_rate_limit(err: Exception) -> bool:
    text = f"{type(err).__name__} {err}".lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


# Groq's 429 body states how long to wait, e.g.
#   "Rate limit reached ... Please try again in 21.9s"
_RETRY_AFTER_RE = re.compile(r"try again in ([0-9.]+)\s*s", re.IGNORECASE)


def retry_after_seconds(err: Exception) -> float | None:
    """The provider's own advice on how long to wait, if it gave any."""
    match = _RETRY_AFTER_RE.search(str(err))
    if match:
        return float(match.group(1))
    headers = getattr(getattr(err, "response", None), "headers", None)
    if headers:
        raw = headers.get("retry-after")
        if raw:
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def build_prompt(question: str, docs: List[Document], history: List[Dict]) -> str:
    return QA_TEMPLATE.format(
        context=format_context(docs),
        chat_history=format_history(history),
        question=question,
    )


def stream_answer(
    llm, prompt: str, retries: int = 3, backoff: float = 2.0, max_wait: float = 65.0
) -> Iterator[str]:
    """
    Yield answer text incrementally.

    Retries transient rate-limit failures before the first token is emitted.
    Once tokens have been yielded a retry would duplicate text, so a mid-stream
    failure is raised to the caller instead.

    The waits are sized to the limit that actually binds here.  Groq's free tier
    caps TOKENS PER MINUTE (6,000) rather than requests per day (14,400), and
    that window refills over as much as 60 seconds.  A 2s/4s backoff therefore
    gave up after six seconds and showed the user a rate-limit error while the
    budget was still refilling.  The provider states how long to wait in its 429
    body, so honour that when present, bounded by max_wait.
    """
    attempt = 0
    while True:
        started = False
        try:
            for chunk in llm.stream(prompt):
                text = getattr(chunk, "content", None) or ""
                if text:
                    started = True
                    yield text
            return
        except Exception as err:  # noqa: BLE001 - provider errors are untyped
            if started or attempt >= retries or not _is_rate_limit(err):
                if _is_rate_limit(err):
                    raise RateLimited(str(err)) from err
                raise
            attempt += 1
            advised = retry_after_seconds(err)
            wait = min(
                advised if advised is not None else backoff * (2 ** (attempt - 1)),
                max_wait,
            )
            log.warning(
                "rate limited (tokens/min), retry %d/%d in %.1fs%s",
                attempt,
                retries,
                wait,
                " (provider-advised)" if advised is not None else "",
            )
            time.sleep(wait)


def answer(
    llm, question: str, store, history: List[Dict] | None = None
) -> Tuple[str, List[Document]]:
    """Non-streaming answer.  Used by the evaluation harness and by tests."""
    history = history or []
    docs = retrieve(store, build_retrieval_query(question, history))
    prompt = build_prompt(question, docs, history)
    t0 = time.time()
    result = llm.invoke(prompt)
    text = getattr(result, "content", str(result))
    log.info("answered chars=%d docs=%d latency=%.2fs", len(text), len(docs), time.time() - t0)
    return text, docs
