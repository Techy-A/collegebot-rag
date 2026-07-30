"""
app.py  --  CollegeBot : grounded college assistant (Streamlit UI)
==================================================================
This file is presentation only.  Retrieval, prompting and generation live in
rag.py so the evaluation harness exercises the same code path the user sees.

Two rules this UI is built around:

1.  Never render model or document text as HTML.  Message bodies go through
     st.markdown() without unsafe_allow_html, so a question containing "<" or a
     PDF chunk containing markup cannot inject anything.  Chrome that does use
     unsafe_allow_html interpolates only html.escape()'d values.
2.  Never assert something the system did not do.  The status pill reflects a
     real health check, the model chip names the model that actually answered,
     a refusal is styled as a refusal, and heuristic scores are labelled as
     heuristics rather than presented as verified metrics.
"""

import html
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import rag
from llm_factory import LABELS, available_models, get_llm
from prompts import REFUSAL_TEXT, is_refusal

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
rag.configure_logging()

st.set_page_config(
    page_title="CollegeBot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit validates avatars against its emoji list, so decorative glyphs like
# "◆" are rejected outright.  Rather than bolt an emoji onto a deliberately
# restrained interface, use Streamlit's built-in monochrome icons and style them
# from assets/style.css.
SUGGESTIONS = [
    "What are the B.Tech admission requirements?",
    "What is the fee structure for B.Tech?",
    "What hostel facilities are available?",
    "What are the library opening hours?",
    "What scholarships can I apply for?",
    "What were the placement figures?",
]


# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
@st.cache_data
def load_css() -> str:
    path = Path(__file__).parent / "assets" / "style.css"
    return path.read_text(encoding="utf-8") if path.exists() else ""


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Resources
# ------------------------------------------------------------------
# Cached on the store path alone.  Model choice and temperature are NOT part of
# the key, so changing them never reloads the index; and no conversation state
# is created in here, so nothing is shared between sessions.
@st.cache_resource(show_spinner=False)
def get_store(faiss_path: str):
    return rag.load_corpus(faiss_path)


FAISS_PATH = os.getenv("FAISS_PATH", "./faiss_store")

store, store_error = None, None
try:
    with st.spinner("Warming up the knowledge base (first load only)..."):
        store = get_store(FAISS_PATH)
except Exception as exc:  # noqa: BLE001 - surfaced to the user below
    store_error = str(exc)

models = available_models()
healthy = store is not None and bool(models)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="cb-brand">CollegeBot</div><div class="cb-brand-sub">Grounded Answers</div>',
        unsafe_allow_html=True,
    )

    if healthy:
        st.markdown(
            '<span class="cb-pill cb-pill-ok"><span class="cb-dot"></span>Online</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="cb-pill cb-pill-down"><span class="cb-dot"></span>Unavailable</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if models:
        model_choice = st.selectbox(
            "Model",
            options=models,
            format_func=lambda m: LABELS.get(m, m),
            help="Only backends that are actually configured are listed.",
        )
    else:
        model_choice = None
        st.warning("No LLM backend configured. Set GROQ_API_KEY in .env")

    show_sources = st.toggle("Show sources", value=True)

    with st.expander("Advanced"):
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            0.05,
            0.05,
            help="Keep low for factual accuracy. Does not reload the index.",
        )
        show_metrics = st.toggle(
            "Heuristic self-check",
            value=False,
            help="Fast keyword-overlap approximations. Not verified metrics.",
        )
        mode = "hybrid dense+BM25 (RRF)" if rag.USE_HYBRID else "dense MMR"
        st.caption(f"Retrieval: {mode}, k={rag.RETRIEVAL_K}, fetch_k={rag.RETRIEVAL_FETCH_K}")
        if rag.USE_RERANKER:
            st.caption("Cross-encoder reranking: on")

    st.markdown("---")

    if st.button("New conversation", use_container_width=True):
        # Session-scoped only.  Never st.cache_resource.clear(), which is
        # process-global and would evict the index for every other visitor.
        st.session_state.messages = []
        st.session_state.pending = None
        st.rerun()

    if store is not None:
        st.markdown(
            f'<div class="cb-footnote">{store.ntotal:,} indexed passages<br>'
            "FAISS + BM25 &middot; MiniLM-L6-v2<br>hybrid RRF retrieval</div>",
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------
# Masthead
# ------------------------------------------------------------------
model_label = LABELS.get(model_choice, model_choice or "not configured")
st.markdown(
    f"""
<div class="cb-masthead">
  <div>
    <div class="cb-masthead-title">CollegeBot</div>
    <div class="cb-masthead-sub">Answers grounded in official college documents &mdash; with citations</div>
  </div>
  <span class="cb-chip">{html.escape(str(model_label))}</span>
</div>
""",
    unsafe_allow_html=True,
)

if store_error:
    st.markdown(
        '<div class="cb-notice cb-notice-error"><span class="cb-notice-icon">◆</span><div>'
        '<span class="cb-notice-label">Knowledge base unavailable</span>'
        f"{html.escape(store_error)}</div></div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# Rendering helpers
# ------------------------------------------------------------------
def render_citations(cards: list) -> None:
    if not cards:
        return
    st.markdown('<div class="cb-cite-head">Sources</div>', unsafe_allow_html=True)
    for c in cards:
        page = f" &middot; page {c['page']}" if c.get("page") else ""
        year = f" &middot; {c['year']}" if c.get("year") else ""
        label = f"[{c['n']}] {c['title']}" + (f" — page {c['page']}" if c.get("page") else "")
        with st.expander(label):
            st.markdown(
                f'<div class="cb-cite-file">'
                f'<span class="cb-cite-n">{c["n"]}</span> '
                f'{html.escape(c["file"])}<span class="cb-cite-page">{page}{year}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="cb-snippet">{html.escape(c["snippet"])}</div>',
                unsafe_allow_html=True,
            )


def render_metrics(scores: dict) -> None:
    if not scores:
        return
    tiles = "".join(
        f'<div class="cb-metric"><div class="cb-metric-v">{scores.get(k, 0):.0%}</div>'
        f'<div class="cb-metric-l">{lbl}</div>'
        f'<div class="cb-metric-bar"><div class="cb-metric-fill" '
        f'style="width:{min(100, scores.get(k, 0) * 100):.0f}%"></div></div></div>'
        for k, lbl in (
            ("faithfulness", "Grounding"),
            ("answer_relevance", "Relevance"),
            ("context_precision", "Precision"),
        )
    )
    st.markdown(f'<div class="cb-metrics">{tiles}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cb-caveat">Fast keyword-overlap heuristics for sanity-checking '
        "only &mdash; not verified RAGAS metrics.</div>",
        unsafe_allow_html=True,
    )


def render_meta(msg: dict) -> None:
    bits = []
    if msg.get("latency"):
        bits.append(f"<b>{msg['latency']:.2f}s</b> response")
    if msg.get("n_sources"):
        bits.append(f"<b>{msg['n_sources']}</b> passages retrieved")
    if msg.get("model"):
        bits.append(f"<b>{html.escape(str(msg['model']))}</b>")
    if bits:
        st.markdown(
            f'<div class="cb-meta">{"".join(f"<span>{b}</span>" for b in bits)}</div>',
            unsafe_allow_html=True,
        )


def render_assistant(msg: dict) -> None:
    """Render one stored assistant turn: notice / answer, citations, meta."""
    if msg.get("kind") == "error":
        st.markdown(
            '<div class="cb-notice cb-notice-error"><span class="cb-notice-icon">◆</span><div>'
            f'<span class="cb-notice-label">{html.escape(msg.get("label", "Error"))}</span>'
            f"{html.escape(msg['content'])}</div></div>",
            unsafe_allow_html=True,
        )
        return

    if msg.get("kind") == "refusal":
        st.markdown(
            '<div class="cb-notice cb-notice-refusal"><span class="cb-notice-icon">◇</span><div>'
            '<span class="cb-notice-label">Not in the knowledge base</span>'
            f"{html.escape(msg['content'])}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(msg["content"])

    if show_sources and msg.get("sources"):
        render_citations(msg["sources"])
    if show_metrics and msg.get("scores"):
        render_metrics(msg["scores"])
    render_meta(msg)


# ------------------------------------------------------------------
# Empty state
# ------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        '<div class="cb-hero">'
        '<div class="cb-hero-mark">collegebot</div>'
        '<div class="cb-hero-sub">Ask about admissions, fees, hostel, courses, '
        "scholarships, placements or campus policy.</div>"
        '<div class="cb-hero-rule"></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if healthy:
        cols = st.columns(3)
        for i, s in enumerate(SUGGESTIONS):
            if cols[i % 3].button(s, key=f"sug{i}", use_container_width=True):
                st.session_state.pending = s
                st.rerun()


# ------------------------------------------------------------------
# History
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            render_assistant(msg)


# ------------------------------------------------------------------
# Input and generation
# ------------------------------------------------------------------
typed = st.chat_input("Ask about admissions, fees, hostel, courses...", disabled=not healthy)
question = typed or st.session_state.pending
st.session_state.pending = None

if question and healthy:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        record = {"role": "assistant", "model": model_label}
        t0 = time.time()
        try:
            with st.spinner("Searching the documents..."):
                llm = get_llm(model_choice, temperature)
                docs = rag.retrieve(store, rag.build_retrieval_query(question, history))
                prompt = rag.build_prompt(question, docs, history)

            answer = st.write_stream(rag.stream_answer(llm, prompt))
            answer = (answer or "").strip()
            latency = time.time() - t0

            if is_refusal(answer) or not answer:
                record.update(
                    kind="refusal",
                    content=answer or REFUSAL_TEXT,
                    sources=[],
                    n_sources=len(docs),
                    latency=latency,
                )
                # The streamed text is already painted as a plain answer.  Store
                # the record first, then rerun so history repaints it with the
                # refusal treatment -- an honest "I don't know" must not look
                # like an answer.  Appending BEFORE the rerun matters:
                # st.rerun() raises immediately, so the append at the end of
                # this block would never execute and the turn would be lost.
                st.session_state.messages.append(record)
                st.rerun()
            else:
                cards = rag.source_cards(docs)
                scores = {}
                if show_metrics:
                    from evaluation.quick_score import quick_evaluate

                    scores = quick_evaluate(question, answer, docs)

                record.update(
                    kind="answer",
                    content=answer,
                    sources=cards,
                    n_sources=len(docs),
                    scores=scores,
                    latency=latency,
                )
                if show_sources:
                    render_citations(cards)
                if show_metrics:
                    render_metrics(scores)
                render_meta(record)

        except rag.RateLimited as exc:
            # The binding limit is tokens-per-minute, not a daily quota, so this
            # clears on its own in under a minute.  Say that, and say how long.
            wait = rag.retry_after_seconds(exc)
            when = f"about {wait:.0f} seconds" if wait else "under a minute"
            record.update(
                kind="error",
                label="Too many questions per minute",
                content=(
                    f"The free Groq tier allows 6,000 tokens per minute and this "
                    f"conversation just used them up. Try again in {when} \u2014 "
                    "nothing is broken and no daily quota has been spent."
                ),
            )
            render_assistant(record)
        except OSError as exc:
            record.update(kind="error", label="Not configured", content=str(exc))
            render_assistant(record)
        except Exception as exc:  # noqa: BLE001
            rag.log.exception("generation failed")
            record.update(
                kind="error",
                label="Something went wrong",
                content=f"{type(exc).__name__}: {exc}",
            )
            render_assistant(record)

    st.session_state.messages.append(record)

st.markdown(
    '<div class="cb-disclaimer">Unofficial project, not affiliated with any institution. '
    "Answers are generated from indexed documents and may be incomplete or out of date "
    "&mdash; verify fees, dates and deadlines with the relevant college office.</div>",
    unsafe_allow_html=True,
)
