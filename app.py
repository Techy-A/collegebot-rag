"""
app.py  --  CollegeBot : Production RAG Chatbot (Streamlit Entry Point)
=========================================================================
"The purpose of abstraction is not to be vague, but to create a new
 semantic level in which one can be absolutely precise."
    -- Edsger W. Dijkstra

Vector store backend: FAISS (replaces ChromaDB for Windows compatibility).
FAISS requires no C++ compiler, no server, and ships pre-built wheels
on every platform including Windows.
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

# Try loading .env from multiple locations.
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
load_dotenv(override=True)
st.set_page_config(
    page_title            = "CollegeBot -- RAG Assistant",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --c-primary    : #4F46E5;
    --c-primary-d  : #3730A3;
    --c-accent     : #059669;
    --c-accent-d   : #065F46;
    --c-bg         : #09090B;
    --c-surface    : #18181B;
    --c-raised     : #27272A;
    --c-text-hi    : #FAFAFA;
    --c-text-mid   : #A1A1AA;
    --c-text-lo    : #52525B;
    --c-border     : #3F3F46;
    --r-sm         : 6px;
    --r-md         : 12px;
    --f-mono       : 'IBM Plex Mono', 'Courier New', monospace;
    --f-sans       : 'IBM Plex Sans', 'Segoe UI', sans-serif;
}
html, body, [data-testid="stAppViewContainer"] {
    background : var(--c-bg); color: var(--c-text-hi); font-family: var(--f-sans);
}
[data-testid="stSidebar"] { background:#0C0C0E; border-right:1px solid var(--c-border); }
[data-testid="stSidebar"] * { color: var(--c-text-hi) !important; }
[data-testid="stSidebar"] label {
    font-family:var(--f-mono); font-size:0.72rem;
    color:var(--c-text-mid) !important; letter-spacing:0.07em; text-transform:uppercase;
}
.cb-topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding:1rem 1.4rem; background:var(--c-surface);
    border:1px solid var(--c-border); border-radius:var(--r-md); margin-bottom:1.4rem;
}
.cb-title { font-family:var(--f-mono); font-size:1.1rem; font-weight:600; color:var(--c-text-hi); }
.cb-subtitle { font-size:0.77rem; color:var(--c-text-mid); margin-top:2px; }
.badge { display:inline-block; border-radius:999px; padding:3px 10px; font-size:0.68rem; font-family:var(--f-mono); font-weight:600; }
.badge-on  { background:#052E16; color:#6EE7B7; border:1px solid #059669; }
.badge-mdl { background:#1E1B4B; color:#A5B4FC; border:1px solid #4F46E5; }
.cb-row { display:flex; align-items:flex-start; gap:10px; margin-bottom:1rem; }
.cb-row-u { flex-direction:row-reverse; }
.cb-av {
    width:34px; height:34px; border-radius:50%; display:flex;
    align-items:center; justify-content:center;
    font-family:var(--f-mono); font-size:0.65rem; font-weight:600; flex-shrink:0;
}
.cb-av-u { background:var(--c-primary-d); color:#C7D2FE; }
.cb-av-b { background:var(--c-accent-d);  color:#6EE7B7; }
.cb-bbl { max-width:75%; padding:0.85rem 1.1rem; border-radius:var(--r-md); font-size:0.91rem; line-height:1.72; }
.cb-bbl-u { background:var(--c-primary); color:#EEF2FF; border-bottom-right-radius:3px; }
.cb-bbl-b { background:var(--c-surface); color:var(--c-text-hi); border:1px solid var(--c-border); border-bottom-left-radius:3px; }
.src-tag {
    display:inline-block; background:var(--c-bg); border:1px solid var(--c-primary-d);
    color:#A5B4FC; font-size:0.67rem; font-family:var(--f-mono);
    padding:2px 7px; border-radius:var(--r-sm); margin:4px 3px 0 0;
}
.mc-row { display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }
.mc { flex:1; background:var(--c-bg); border:1px solid var(--c-border); border-radius:var(--r-sm); padding:8px 14px; text-align:center; }
.mc-v { font-family:var(--f-mono); font-size:1.25rem; font-weight:600; color:#6EE7B7; }
.mc-l { font-size:0.65rem; color:var(--c-text-mid); margin-top:2px; font-family:var(--f-mono); text-transform:uppercase; }
.lat { font-size:0.65rem; color:var(--c-text-lo); font-family:var(--f-mono); margin-top:6px; }
[data-testid="stChatInput"]>div { background:var(--c-surface) !important; border:1px solid var(--c-border) !important; border-radius:var(--r-md) !important; }
[data-testid="stChatInput"] textarea { color:var(--c-text-hi) !important; }
.stButton>button {
    background:var(--c-surface); color:var(--c-text-hi); border:1px solid var(--c-border);
    border-radius:var(--r-sm); font-family:var(--f-sans); font-size:0.82rem; transition:border-color 0.15s;
}
.stButton>button:hover { background:var(--c-raised); border-color:var(--c-primary); color:#A5B4FC; }
hr { border-color:var(--c-border); }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:var(--c-border); border-radius:2px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Pipeline loader -- uses FAISS instead of ChromaDB.
# FAISS MMR is accessed via max_marginal_relevance_search() wrapped
# inside a custom retriever so it plugs into the LangChain chain API.
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Initialising RAG pipeline...")
def load_pipeline(llm_choice: str, temperature: float):
    """
    Build the ConversationalRetrievalChain backed by FAISS.

    "Make it work, make it right, make it fast -- in that order."
        -- Kent Beck
    """
    # Force load .env from the project root before anything else runs.
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
    
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
    from llm_factory import get_llm

    # --- Embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name    = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )

    # --- FAISS vector store ---
    faiss_path = os.getenv("FAISS_PATH", "./faiss_store")
    if not os.path.isdir(faiss_path):
        raise FileNotFoundError(
            f"No FAISS store found at '{faiss_path}'.  "
            "Run: python ingest.py --sample"
        )
    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True,   # safe: we wrote this file ourselves
    )

    # --- MMR retriever ---
    # FAISS exposes MMR via as_retriever with search_type="mmr".
    # Optimized for better context precision: increased fetch_k to 30
    retriever = vectorstore.as_retriever(
        search_type   = "mmr",
        search_kwargs = {"k": 6, "fetch_k": 30, "lambda_mult": 0.5},
    )

    # --- Grounding prompt ---
    # Optimized for better faithfulness and answer relevance
    QA_PROMPT = PromptTemplate(
        template="""You are CollegeBot, a precise and reliable assistant for college
students, faculty, and administrative staff.

STRICT RULES:
1. Answer ONLY from the provided CONTEXT below. Every sentence in your answer 
   must come directly from the CONTEXT or be a direct paraphrase of it.
2. Directly address the question first before providing additional details.
3. If the context does not contain the answer, respond with exactly:
   "I do not have that information in my knowledge base. Please contact
    the relevant college office directly."
4. Never invent deadlines, fee amounts, names, dates, or policy details.
5. Use bullet points for any list of three or more items.
6. Cite the source document name when it is present in the metadata.

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

STUDENT QUESTION: {question}

ANSWER:""",
        input_variables=["context", "chat_history", "question"],
    )

    # --- Memory ---
    memory = ConversationBufferWindowMemory(
        k              = 5,
        memory_key     = "chat_history",
        return_messages= True,
        output_key     = "answer",
    )

    # --- LLM ---
    llm = get_llm(llm_choice, temperature)

    # --- Chain ---
    chain = ConversationalRetrievalChain.from_llm(
        llm                       = llm,
        retriever                 = retriever,
        memory                    = memory,
        combine_docs_chain_kwargs = {"prompt": QA_PROMPT},
        return_source_documents   = True,
        verbose                   = False,
    )
    return chain, vectorstore


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.95rem;'
        'font-weight:600;color:#FAFAFA;padding:0.4rem 0;">CollegeBot</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<span class="badge badge-on">ONLINE</span>', unsafe_allow_html=True)
    st.markdown("---")

    llm_choice = st.selectbox(
        "Language Model",
        options=[
            "groq/llama-3.1-8b-instant",
            "phi3-mini-finetuned",
            "mistral-7b-finetuned",
        ],
        help="Groq is recommended -- free, no GPU needed, fast.",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.05, 0.05,
                            help="Keep at 0.05 for maximum factual accuracy.")
    show_sources = st.toggle("Show retrieved sources", value=True)
    show_metrics = st.toggle("Show inline eval scores",  value=False)
    st.markdown("---")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.cache_resource.clear()
        st.rerun()

    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;'
        'color:#3F3F46;margin-top:1.2rem;line-height:1.8;">'
        'RAG -- MMR -- QLoRA<br>FAISS -- all-MiniLM-L6-v2<br>RAGAS evaluation</div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Top bar
# ------------------------------------------------------------------
model_label = {
    "groq/llama-3.1-8b-instant": "Groq / Llama-3.1-8B",
    "phi3-mini-finetuned"       : "Phi-3-mini  (fine-tuned)",
    "mistral-7b-finetuned"      : "Mistral-7B  (fine-tuned)",
}.get(llm_choice, llm_choice)

st.markdown(f"""
<div class="cb-topbar">
  <div>
    <div class="cb-title">CollegeBot</div>
    <div class="cb-subtitle">Production RAG assistant -- FAISS backend</div>
  </div>
  <span class="badge badge-mdl">{model_label}</span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------------------------------------------------
# Pipeline load
# ------------------------------------------------------------------
try:
    chain, _vs  = load_pipeline(llm_choice, temperature)
    pipeline_ok = True
except FileNotFoundError as e:
    st.error(str(e))
    pipeline_ok = False
except Exception as e:
    st.error(
        f"Pipeline failed to initialise: {e}  "
        "Verify GROQ_API_KEY is set in .env and all packages are installed."
    )
    pipeline_ok = False

# ------------------------------------------------------------------
# Render chat history
# ------------------------------------------------------------------
with st.container():
    if not st.session_state.messages:
        st.markdown("""
<div style="text-align:center;padding:3rem 0 1.5rem;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;color:#27272A;letter-spacing:-2px;">collegebot</div>
  <div style="font-size:0.82rem;color:#52525B;margin-top:0.5rem;">
    Ask about admissions, fees, courses, hostel, scholarships, or campus policy.
  </div>
</div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        prompts = [
            "What are the admission requirements?",
            "What scholarships are available?",
            "How do I register for elective courses?",
        ]
        for col, prompt in zip([c1, c2, c3], prompts):
            if col.button(prompt, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
<div class="cb-row cb-row-u">
  <div class="cb-av cb-av-u">YOU</div>
  <div class="cb-bbl cb-bbl-u">{msg["content"]}</div>
</div>""", unsafe_allow_html=True)
        else:
            src_html = ""
            if show_sources and msg.get("sources"):
                tags = "".join(f'<span class="src-tag">{s}</span>' for s in msg["sources"][:5])
                src_html = f'<div style="margin-top:8px">{tags}</div>'

            met_html = ""
            if show_metrics and msg.get("scores"):
                sc = msg["scores"]
                met_html = f"""
<div class="mc-row">
  <div class="mc"><div class="mc-v">{sc.get('faithfulness',0):.0%}</div><div class="mc-l">Faithfulness</div></div>
  <div class="mc"><div class="mc-v">{sc.get('answer_relevance',0):.0%}</div><div class="mc-l">Answer Relevance</div></div>
  <div class="mc"><div class="mc-v">{sc.get('context_precision',0):.0%}</div><div class="mc-l">Context Precision</div></div>
</div>"""

            lat_html = f'<div class="lat">latency: {msg["latency"]:.3f}s</div>' if msg.get("latency") else ""

            st.markdown(f"""
<div class="cb-row">
  <div class="cb-av cb-av-b">BOT</div>
  <div class="cb-bbl cb-bbl-b">{msg["content"]}{src_html}{met_html}{lat_html}</div>
</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------
if pipeline_ok:
    user_input = st.chat_input("Ask about admissions, fees, courses, hostel, exams...")
    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.rerun()

# ------------------------------------------------------------------
# Response generation
# ------------------------------------------------------------------
if (
    pipeline_ok
    and st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
):
    q = st.session_state.messages[-1]["content"]

    with st.spinner("Retrieving context and generating answer..."):
        try:
            t0      = time.time()
            result  = chain.invoke({"question": q})
            latency = round(time.time() - t0, 3)

            answer  = result.get("answer", "No answer was generated.")
            docs    = result.get("source_documents", [])
            sources = list({
                doc.metadata.get("source", "unknown").split("/")[-1].split("\\")[-1]
                for doc in docs
            })

            scores = {}
            if show_metrics:
                from evaluation.quick_score import quick_evaluate
                scores = quick_evaluate(q, answer, docs)

        except Exception as e:
            answer  = f"Error generating response: {e}"
            sources = []
            scores  = {}
            latency = 0.0

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": answer,
        "sources": sources,
        "scores" : scores,
        "latency": latency,
    })
    st.rerun()
