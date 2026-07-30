# CollegeBot

A grounded question-answering assistant for college information. It answers questions
about admissions, fees, hostels, courses, scholarships, placements and campus policy
**only** from a corpus of source documents, cites the document and page it used, and
declines when the answer is not in the corpus.

Runs entirely on free infrastructure: FAISS (vector store), Hugging Face
sentence-transformers (embeddings), Groq Cloud (inference), Streamlit (UI).

**Current index:** 2,185 chunks from 24 documents - 16 institutional PDFs plus 8
researched reference documents on Thapar Institute (TIET), Patiala.

---

## Table of contents

- [How to run](#how-to-run)
- [System architecture](#system-architecture)
- [Files](#files)
- [Design decisions](#design-decisions-worth-knowing)
- [Evaluation](#evaluation)
- [Fine-tuning (QLoRA)](#fine-tuning-qlora)
- [Configuration](#configuration)
- [Deploying](#deploying-to-streamlit-community-cloud)
- [Known limitations](#known-limitations)

---

## How to run

### 1. Prerequisites

- **Python 3.9+** (3.11 recommended - 3.9 reached end of life in October 2025)
- A **free Groq API key** from [console.groq.com/keys](https://console.groq.com/keys)
  (no credit card required)
- ~1 GB disk for dependencies, ~10 MB for the index

### 2. Install

```bash
cd collegebot-rag
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API key

```bash
cp .env.example .env
```

Then edit `.env` so it contains at minimum:

```
GROQ_API_KEY=gsk_your_key_here
FAISS_PATH=./faiss_store
```

`.env` is git-ignored. Never commit it.

### 4. Build the search index

```bash
python ingest.py
```

Reads everything in `data/`, splits it into 800-character chunks, embeds them on CPU and
writes `faiss_store/`. Takes about a minute for the bundled corpus and prints a retrieval
smoke test at the end.

No documents of your own? `python ingest.py --sample` writes a small synthetic handbook
first, so the pipeline is testable immediately.

### 5. Run the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**. The first question of a session takes a few seconds
longer while the embedding model and index load - the UI says so explicitly.

### 6. Try it

Ask something the corpus covers, e.g.:

- *"What is the B.Tech tuition fee?"*
- *"How many hostels are there and what do they cost?"*
- *"What are the library opening hours?"*
- *"What was the average placement package?"*

Then ask something it should refuse, e.g. *"What is the B.Tech fee at IIT Delhi?"* - a
refusal renders as a distinct amber notice, never as an answer.

### Other commands

```bash
pytest tests/ -q                                  # 64 tests, no API key or index needed
python evaluation/ragas_eval.py                   # grounding checks on the real corpus
python evaluation/ragas_eval.py --ragas --limit 8 # + LLM-judged RAGAS on a subset
python evaluation/ragas_eval.py --multi-turn      # follow-up conversation scenarios
python generate_dataset.py --mode curated         # build the QLoRA training set
ruff check . && ruff format --check .             # lint
```

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No FAISS store found at './faiss_store'` | Run `python ingest.py` |
| Sidebar shows **Unavailable**, "No LLM backend configured" | `GROQ_API_KEY` missing from `.env` |
| *"Too many questions per minute"* notice | Groq's free tier allows **6,000 tokens/minute**. It refills in under a minute; nothing is broken and no daily quota is spent. |
| First answer is slow | One-time embedding model + index load per session |
| `Port 8501 is already in use` | `pkill -f "streamlit run app.py"`, then start again |
| Fine-tuned models missing from the dropdown | Correct behaviour - they only appear when an endpoint is actually configured. See [Fine-tuning](#fine-tuning-qlora). |

---

## System architecture

```
┌─────────────────────────── INGEST (offline, one command) ──────────────────────────┐
│                                                                                    │
│   data/*.pdf *.txt *.docx *.html                                                   │
│         │                                                                          │
│         ├─ load          PyPDFLoader / TextLoader / Docx2txt / UnstructuredHTML    │
│         ├─ dedupe        SHA-256 digest, skip byte-identical files                 │
│         ├─ tag year      "fee-structure-2026.pdf" → effective_year = 2026          │
│         ├─ split         RecursiveCharacterTextSplitter, 800 chars / 150 overlap    │
│         ├─ embed         all-MiniLM-L6-v2, 384-dim, CPU                            │
│         └─ persist       faiss_store/index.faiss + index.pkl + manifest.json       │
│                                                       (ingest.py)                  │
└────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── QUERY (live, per question) ─────────────────────────────┐
│                                                                                    │
│   user question ──┬─────────────────────────────────────────────────────┐          │
│                   │                                                     │          │
│                   ▼                                                     │          │
│      build_retrieval_query()                                            │          │
│      prepends the previous question so elliptical follow-ups            │          │
│      ("and for M.Tech?") stay retrievable - costs no extra LLM call     │          │
│                   │                                                     │          │
│         ┌─────────┴─────────┐                                           │          │
│         ▼                   ▼                                           │          │
│   FAISS dense          BM25 sparse                                      │          │
│   (top 30)             (top 30)                                         │          │
│         └─────────┬─────────┘                                           │          │
│                   ▼                                                     │          │
│      Reciprocal Rank Fusion  →  top k=8 passages                        │          │
│      (measured: 77.1% → 88.6% fact-in-context vs dense MMR alone)       │          │
│                   │                                                     │          │
│                   ▼                                                     ▼          │
│      build_prompt()  ── numbered passages + source + page + YEAR ── conversation   │
│                   │                                     grounding rules   history  │
│                   ▼                                     (prompts.py)               │
│      Groq llama-3.1-8b-instant, temperature 0.05, streamed                         │
│      retry honours the provider's own "try again in Ns" hint                        │
│                   │                                                                │
│                   ▼                                                                │
│      is_refusal()? ──── yes ──▶ amber "not in the knowledge base" notice            │
│                   │                                                                │
│                   no                                                               │
│                   ▼                                                                │
│      streamed answer + numbered citation cards (title, page, year, snippet)         │
│                     + meta line (latency, passages retrieved, model)               │
│                                              (rag.py → app.py)                     │
└────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── EVALUATION (offline) ───────────────────────────────────┐
│                                                                                    │
│   evaluation/gold_set.py    39 hand-written Q&A over the REAL corpus,               │
│                             including 4 that must be refused, + multi-turn          │
│         │                                                                          │
│         ├─ Layer 1  deterministic, no API calls                                    │
│         │           refusal accuracy · fact recall · per-domain breakdown           │
│         │                                                                          │
│         └─ Layer 2  opt-in --ragas, judged by llama-3.3-70b-versatile              │
│                     faithfulness · answer relevance · context precision            │
│                              (evaluation/ragas_eval.py)                            │
│         ▼                                                                          │
│   evaluation/eval_results.json  - scores PLUS corpus path, retrieval config,        │
│                                   judge model and every per-question outcome        │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Request lifecycle in one line

`question → (+prev question) → dense∪sparse → RRF → top-8 → grounded prompt → Groq (streamed) → answer + citations`, or a styled refusal.

### Process model

`st.cache_resource` holds **only** the embedding model, FAISS index and BM25 index -
stateless objects, safe to share across every visitor to the deployment. Conversation
state lives in `st.session_state`, per session. Getting this boundary wrong is what
caused the cross-session conversation bleed described below.

---

## Files

### Application

| File | Lines | Role |
|---|---:|---|
| `app.py` | 414 | Streamlit UI **only** - layout, chat rendering, citation cards, refusal/error states. No retrieval logic. |
| `rag.py` | 379 | The pipeline: corpus loading, hybrid retrieval, RRF, prompt assembly, streaming, rate-limit retry, logging. Importable without Streamlit. |
| `prompts.py` | 51 | The grounding prompt and refusal text. Single source of truth for app **and** evaluation, so they cannot drift. |
| `llm_factory.py` | 158 | Backend selection with the honesty contract - only serves what is actually configured. |
| `assets/style.css` | 551 | The premium black theme. |

### Data pipeline

| File | Lines | Role |
|---|---:|---|
| `ingest.py` | 383 | Documents → dedupe → year tag → chunk → embed → FAISS + manifest. |
| `data/` | 24 files | 16 institutional PDFs + 8 researched Thapar reference documents (each ending in its source URLs). |
| `faiss_store/` | - | `index.faiss`, `index.pkl`, and `manifest.json` recording build time, embedding model, chunk settings and per-file years. |

### Evaluation

| File | Lines | Role |
|---|---:|---|
| `evaluation/gold_set.py` | 321 | 39 gold Q&A over the real corpus + 3 multi-turn scenarios. Each item carries `must_include` tokens and an `expect_refusal` flag. |
| `evaluation/ragas_eval.py` | 381 | Two-layer harness. Deterministic grounding checks always run; RAGAS is opt-in. |
| `evaluation/quick_score.py` | 167 | Sub-10 ms keyword heuristics for the optional inline display. Labelled as heuristics in the UI, never as verified metrics. |
| `evaluation/eval_results.json` | - | Committed on purpose: a metric nobody can reproduce is not evidence. |

### Fine-tuning

| File | Lines | Role |
|---|---:|---|
| `curated_dataset.py` | 435 | Hand-authored training set: 55 grounded pairs + 15 refusal pairs, each carrying its grounding context. |
| `generate_dataset.py` | 448 | Dataset CLI - `--mode curated` (recommended), `--mode groq`, `--mode local`. |
| `notebooks/CollegeBot_Colab.py` | - | QLoRA cells for Phi-3-mini and Mistral-7B on a free Colab T4. |

### Tests and tooling

| File | Lines | Role |
|---|---:|---|
| `tests/test_rag.py` | 352 | Retrieval, RRF, citations, prompt assembly, streaming, rate-limit retry. |
| `tests/test_factory_and_ingest.py` | 161 | The honesty contract, year tagging, digest dedup, heuristic scorer. |
| `tests/test_dataset_generation.py` | 63 | The Q&A parser, including the multi-line truncation regression. |
| `.github/workflows/ci.yml` | - | ruff lint + format check + pytest on every push. |
| `pyproject.toml` | - | ruff and pytest configuration. |

All 64 tests run offline: no API key, no index, no network.

### Documentation

| File | Role |
|---|---|
| `README.md` | This file. |
| `SUMMARY_REPORT.md` | What changed and why, with measured before/after numbers and honest limitations. |
| `council-improvements.md` | The multi-perspective review that produced the work plan. |
| `LICENSE` | MIT, plus a note that `data/` is third-party material not covered by it. |

---

## Design decisions worth knowing

### The UI never asserts something the system did not do

- The **status pill** reflects a real health check (index loaded *and* a backend
  configured), not a hardcoded literal.
- The **model chip** names the model that actually answered.
- A **refusal is styled as a refusal.** An honest "I don't have that" must not look like
  an answer.
- **Heuristic scores are labelled as heuristics**, not presented as verified RAGAS metrics.
- Message bodies render through `st.markdown()` **without** `unsafe_allow_html`, so
  neither a question containing `<` nor a PDF chunk containing markup can inject anything.

### The LLM factory's honesty contract

`available_models()` returns only backends that can *actually* be served. Groq appears
when `GROQ_API_KEY` is set; a fine-tuned model appears only when its Inference Endpoint
URL **and** `HF_TOKEN` are both configured. `get_llm()` raises rather than quietly
substituting a different model.

This replaces a design that advertised three models, silently fell back to Groq when the
two fine-tuned ones could not load, and kept displaying a "Phi-3-mini (fine-tuned)" badge
over a Groq answer.

### Session isolation

`st.cache_resource` is a **process-global singleton** on Streamlit. An earlier version
constructed the conversation memory *inside* the cached function, so concurrent visitors
shared one mutable buffer and their conversations could bleed into each other's context.
The same bug meant moving the temperature slider re-read the whole index from disk, and
"Clear conversation" called `st.cache_resource.clear()`, evicting the index for every
other visitor. Now: stateless objects cached, conversation state per session.

### Hybrid retrieval, chosen by measurement

| Configuration | Fact reaches context | Added latency |
|---|---|---|
| Dense MMR k=6 (previous default) | 77.1% | - |
| BM25 only | 80.0% | - |
| Dense + cross-encoder reranker | 88.6% | ~2 s/query |
| **Hybrid dense + BM25 (RRF)** | **88.6%** | **none** |
| Hybrid + cross-encoder | 85.7% | ~1.4 s/query |

Hybrid ties the best reranker at zero cost, and stacking a reranker on top made it
*worse*. Sparse retrieval earns its place because this corpus is full of proper nouns,
exact amounts and list headings ("TOP RECRUITERS", "Rs 17,24,000", "A++") where lexical
matching beats a 384-dim dense embedding. Reranking remains available via `USE_RERANKER=1`.

### No `ConversationalRetrievalChain`

That legacy chain fires an extra condense-question LLM call before retrieval on every
follow-up - two sequential requests per turn against a rate-limited tier - and does not
stream cleanly. Retrieve → prompt → stream is cheaper and directly streamable. Follow-ups
stay retrievable by prepending the previous question, which costs no extra API call.

### Document versioning

The corpus holds same-topic documents from different years (a 2025 and a 2026 fee
structure). `ingest.py` tags every chunk with an effective year, `rag.py` puts that year
**in the context the model sees**, and the prompt instructs it to prefer the most recent
source and say which year applies. Citations display the year too.

---

## Evaluation

`evaluation/ragas_eval.py` runs the gold set against the **real** indexed corpus using the
exact prompt and retrieval config the app serves - both imported, so they cannot drift.

**Layer 1 - grounding checks** (deterministic, no API calls, always runs)

- `refusal_accuracy` - did it decline exactly when it should have? A **false refusal on an
  in-corpus question** is a real failure and is reported by name.
- `fact_recall` - do answers contain the key figures from the source document?

**Layer 2 - RAGAS** (opt-in, `--ragas`): faithfulness / answer relevance / context
precision, judged by `llama-3.3-70b-versatile` - a stronger model than the one under test,
because using the same 8B to grade its own output invites self-preference bias. Refusals
are excluded from RAGAS averages, since a correct "I don't know" scores near zero on
faithfulness and would mask real generation quality.

Latest measured numbers, the honest caveats, and the diagnosis of what still fails are in
[SUMMARY_REPORT.md](SUMMARY_REPORT.md). Short version: **refusal accuracy 0.87–0.97
against a 0.95 target, fact recall 0.74–0.77 against 0.85.** Neither target is reliably
met, and the run-to-run spread was larger than some of the improvements being measured -
which is why the harness is now pinned to temperature 0.0.

---

## Fine-tuning (QLoRA)

The QLoRA pipeline is **built, verified and ready to execute end to end**: the training
set is authored and validated, the Colab notebook is corrected and compiles, and the
serving path is wired and covered by tests.

**Status of the training run itself:** not yet executed. It needs a CUDA GPU (the
notebook targets a free Colab T4) plus a Google account and a write-scope Hugging Face
token, so it is a one-command step the repository owner runs. Until an endpoint exists the
UI deliberately does not offer a fine-tuned option - see
[the honesty contract](#the-llm-factorys-honesty-contract).

### How the fine-tuning works

**1. Method - QLoRA.** The base model is loaded in 4-bit NF4 quantization, its weights
frozen, and small low-rank adapter matrices are trained on top. Only the adapters receive
gradients, which is what makes a 7B model trainable in ~15 GB of T4 VRAM. Double
quantization is applied to the quantization constants for a further VRAM saving, and the
same quantization config is used at training and inference time so there is no
quantization mismatch between them.

**2. Hyperparameters** and why each is set that way:

| Parameter | Value | Reason |
|---|---|---|
| LoRA rank `r` | 16 | Enough capacity for domain adaptation without exhausting T4 VRAM |
| LoRA `alpha` | 32 | Conventional 2× scaling relative to rank |
| Base precision | 4-bit NF4 | Halves VRAM against fp16; NF4 suits normally-distributed weights |
| Batch size | 2 (Phi-3) / 1 (Mistral-7B) | The largest that fits; Mistral is the bigger model |
| Gradient accumulation | 4 / 8 | Gives an effective batch of 8 in both cases |
| Learning rate | 2e-4 | Standard for LoRA on instruction data |
| Epochs | 2 | Domain adaptation without overfitting a small set |
| Gradient checkpointing | on | Trades compute for ~30% VRAM |
| Optimizer | 8-bit AdamW | Keeps optimizer state off the VRAM budget |

**3. Training data.** `curated_dataset.py` holds 70 hand-authored pairs - 55 grounded and
15 refusal - written to the same house style the production prompt enforces. Two
decisions matter more than the count:

- **Every record carries its retrieved context.** Records are Alpaca-style
  `{instruction, input, output}` where `input` is the grounding passage. At inference this
  model never answers from memory - it receives retrieved passages and must answer only
  from them - so training must mirror that. A contextless dataset teaches the model to
  answer from parametric memory, which fights the grounding prompt and produces confident
  wrong answers.
- **Refusals are training data.** 21% of records pair a reasonable question with context
  that genuinely does not answer it: another institution's fees, personal student records,
  prompt-injection attempts, subjective judgements, future predictions. A generator that
  derives questions *from* the passage containing the answer cannot produce these, so it
  cannot teach the one behaviour this product depends on.

**4. Prompt format used for training** - deliberately mirrors serving, and terminates with
EOS so the model learns to stop:

```
### Context:
{retrieved passages}

### Question:
{student question}

### Answer:
{grounded answer}<eos>
```

**5. Steps to run it:**

```bash
python generate_dataset.py --mode curated     # writes dataset/train.jsonl + eval.jsonl
```

1. Upload `dataset/` to Google Drive (and your source PDFs, if rebuilding the index there).
2. Open [colab.research.google.com](https://colab.research.google.com) → **Runtime →
   Change runtime type → T4 GPU**.
3. Add `HF_TOKEN` (write scope) and `GROQ_API_KEY` as **Colab Secrets** - the notebook
   reads them via `userdata.get()`, so tokens never sit in a cell.
4. Copy the cells from `notebooks/CollegeBot_Colab.py` and run them in order. Roughly
   60–75 min for Phi-3-mini and ~90 min for Mistral-7B.
5. The **adapter is pushed first** (a few MB, seconds) so a session timeout cannot lose
   the run; the merged 16-bit model is pushed after, behind a `PUSH_MERGED` flag.
6. Create a Hugging Face Inference Endpoint for the pushed model, then set
   `PHI3_ENDPOINT_URL` (or `MISTRAL_ENDPOINT_URL`) in `.env`. The option appears in the UI
   automatically - no code change.

**6. Always use `--mode curated`.** The two automatic generators produce data that would
make the model worse; the measured comparison is in
[SUMMARY_REPORT.md](SUMMARY_REPORT.md) (36% of generated answers under 60 characters, 16%
truncated mid-phrase, 0% carrying context, 0 refusal examples).

**7. What fine-tuning will and will not fix.** It buys format adherence, house style and
refusal discipline. It does **not** improve factual grounding - that is a retrieval and
source-authority problem, and the current evaluation gap lives there, not in the model
weights. Treat the fine-tune as a behaviour-shaping step, and keep judging factual quality
with `evaluation/ragas_eval.py`.

---

## Configuration

All optional except `GROQ_API_KEY`. Set in `.env` or Streamlit secrets.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | - | **Required.** Free at console.groq.com |
| `FAISS_PATH` | `./faiss_store` | Index location |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Serving model |
| `JUDGE_MODEL` | `llama-3.3-70b-versatile` | RAGAS judge |
| `RETRIEVAL_K` | `8` | Passages passed to the model |
| `RETRIEVAL_FETCH_K` | `30` | Candidate pool per retriever |
| `USE_HYBRID` | `1` | Dense + BM25 fusion; `0` for dense-only MMR |
| `USE_RERANKER` | off | Cross-encoder reranking |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `LOG_LEVEL` | `INFO` | stdout logging |
| `HF_TOKEN`, `PHI3_ENDPOINT_URL`, `MISTRAL_ENDPOINT_URL` | - | Fine-tuned backends |

---

## Deploying to Streamlit Community Cloud

1. Commit `faiss_store/` - the index must ship, there is no GPU to rebuild it.
2. Point a new Streamlit Cloud app at `app.py`.
3. Under **Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key"
   FAISS_PATH   = "./faiss_store"
   ```
4. Free apps sleep after inactivity, so a cold link shows a wake-up screen first.

Never commit `.env` or `.streamlit/secrets.toml`.

---

## Known limitations

- **Neither evaluation target is reliably met** (see above). The main cause is diagnosed:
  large older documents sometimes outrank current ones, so the model can quote a 2018
  figure. Per-document authority weighting is the next measured change.
- **The free Groq tier is the throughput ceiling** - 6,000 tokens/minute, shared across
  everyone using a deployment. Roughly 2–3 questions per minute; the UI explains the wait
  rather than showing a bare error.
- **Answers are only as current as the documents.** Fees, deadlines and rankings change
  every cycle - verify anything consequential with the institution directly.
- **Tabular PDFs are chunked as prose**, so fee tables can split mid-row. Table-aware
  extraction is the next retrieval improvement worth measuring.
- **The corpus is trusted.** There is no structural separation between instructions and
  retrieved text, so do not add unreviewed third-party documents to `data/`.
- **No conversation persistence.** Refreshing the page starts a new session.
- **Python 3.9 is past end of life** (October 2025). The code runs on it; a bump to 3.11+
  is recommended.

---

## Licence

MIT - see [LICENSE](LICENSE). The documents under `data/` are third-party institutional
material and are **not** covered by that licence; verify redistribution rights before
publishing this repository.
