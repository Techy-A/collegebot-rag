# CollegeBot

A grounded question-answering assistant for college information. It answers
questions about admissions, fees, hostels, courses, scholarships, placements and
campus policy **only** from a corpus of source documents, cites the document and
page it used, and declines when the answer is not in the corpus.

Built entirely on free infrastructure: FAISS (vector store), Hugging Face
sentence-transformers (embeddings), Groq Cloud (inference), Streamlit (UI).

```
                        INGEST  (offline, one command)
   data/*.pdf|*.txt ─▶ load ─▶ tag effective year ─▶ split (800 chars, 150 overlap)
                    ─▶ embed with all-MiniLM-L6-v2 (384-dim, CPU)
                    ─▶ FAISS index + manifest.json

                        QUERY  (live, per question)
   question ─▶ (+ previous question, so follow-ups are retrievable)
            ─▶ MMR retrieval: fetch_k=30 candidates → k=6 diverse passages
            ─▶ numbered context + grounding prompt
            ─▶ Groq llama-3.1-8b-instant, streamed token by token
            ─▶ answer + page-numbered citations + latency
```

---

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# free key, no card: https://console.groq.com/keys
echo 'GROQ_API_KEY=gsk_your_key_here' > .env

python ingest.py          # build the index from data/  (~1 min)
streamlit run app.py      # http://localhost:8501
```

```bash
pytest tests/ -q                      # 39 offline tests, no API key needed
python evaluation/ragas_eval.py       # grounding checks against the real corpus
```

---

## What is in the box

| File | Role |
|---|---|
| `app.py` | Streamlit UI only — no retrieval logic |
| `rag.py` | Retrieval, prompt assembly, streaming, retry, logging |
| `prompts.py` | The grounding prompt — single source of truth for app *and* eval |
| `llm_factory.py` | LLM backend selection, with an honesty contract (below) |
| `ingest.py` | Documents → chunks → FAISS, with year tagging and a manifest |
| `evaluation/gold_set.py` | 39 hand-written Q&A pairs against the real corpus |
| `evaluation/ragas_eval.py` | Two-layer evaluation harness |
| `evaluation/quick_score.py` | Sub-10 ms heuristic scores for inline display |
| `assets/style.css` | The premium black theme |
| `tests/` | Offline pytest suite |

---

## Design decisions worth knowing

### The UI never asserts something the system did not do

- The **status pill** reflects a real health check (index loaded *and* a backend
  configured), not a hardcoded "ONLINE".
- The **model chip** names the model that actually answered.
- A **refusal is styled as a refusal.** An honest "I don't have that" must not
  look like an answer.
- The **heuristic scores** are labelled as keyword-overlap heuristics, not
  presented as verified RAGAS metrics.
- Message bodies render through `st.markdown()` **without** `unsafe_allow_html`,
  so neither a question containing `<` nor a PDF chunk containing markup can
  inject anything into the page.

### The LLM factory's honesty contract

`available_models()` returns only backends that can *actually* be served right
now. Groq appears when `GROQ_API_KEY` is set; a fine-tuned model appears only
when its Inference Endpoint URL **and** `HF_TOKEN` are both configured.
`get_llm()` raises rather than quietly substituting a different model.

This replaces an earlier design that advertised three models, silently fell back
to Groq when the two fine-tuned ones could not load, and kept displaying a
"Phi-3-mini (fine-tuned)" badge over a Groq answer. The QLoRA training code that
would produce those adapters is in `notebooks/CollegeBot_Colab.py`; until an
endpoint is deployed, those options simply do not appear in the UI.

### Session isolation

Only the stateless, expensive objects (embedding model, FAISS index) are cached
process-wide via `st.cache_resource`. Conversation state lives in
`st.session_state`.

This matters: `st.cache_resource` is a **process-global singleton** on Streamlit.
An earlier version constructed the conversation memory *inside* the cached
function, so concurrent visitors on the same deployment shared one mutable memory
buffer and their conversations could bleed into each other's context. The same
bug also meant moving the temperature slider re-read the whole index from disk,
and "Clear conversation" called `st.cache_resource.clear()`, evicting the index
for every other visitor.

### No `ConversationalRetrievalChain`

That legacy chain fires an extra condense-question LLM call before retrieval on
every follow-up turn — two sequential requests per turn against a 30 req/min free
tier — and it does not stream the final answer cleanly. Retrieve → prompt →
stream is cheaper and directly streamable. Follow-ups stay retrievable by
prepending the previous question to the retrieval query, which costs no extra
API call.

### Document versioning

The corpus contains same-topic documents from different years (a 2025 and a 2026
fee structure). MMR retrieval *maximises diversity*, so without a year tag it can
return one chunk from each and the model may blend two years' figures. `ingest.py`
tags every chunk with an effective year, the prompt instructs the model to prefer
the most recent and say which year it belongs to, and citations show the year.
`faiss_store/manifest.json` records what went into the index and when, so
staleness is observable instead of silent.

---

## Evaluation

`evaluation/ragas_eval.py` runs the gold set against the **real** indexed corpus
using the exact prompt and retrieval config the app serves (both imported, so
they cannot drift). Two layers:

**Layer 1 — grounding checks (deterministic, no API calls, always runs)**

- `refusal_accuracy` — did it decline exactly when it should have? A **false
  refusal on an in-corpus question** is a real failure and is reported by name.
- `fact_recall` — do answers contain the key figures from the source document?

**Layer 2 — RAGAS (opt-in, `--ragas`)**

`faithfulness` / `answer_relevancy` / `context_precision`, LLM-judged.

Two things make this trustworthy that the previous harness got wrong:

1. **The judge is a stronger model than the system under test**
   (`llama-3.3-70b-versatile` by default). Using the same 8B model as both
   author and grader invites self-preference bias, and claim decomposition is
   exactly what small models are weakest at.
2. **Refusals are excluded from RAGAS averages.** "I don't have that
   information" is correct behaviour but scores near zero on faithfulness and
   relevance; including it would hide real generation quality. Refusal
   correctness is measured properly in layer 1 instead.

RAGAS is opt-in because the free Groq tier allows ~30 requests/minute while RAGAS
issues on the order of a hundred judge sub-calls, so a full run is slow and prone
to rate-limit timeouts that silently become NaN → 0.0. Layer 1 gives an honest
signal on every run.

```bash
python evaluation/ragas_eval.py                    # layer 1, all 39 items
python evaluation/ragas_eval.py --ragas --limit 8  # + RAGAS on a subset
python evaluation/ragas_eval.py --multi-turn       # follow-up scenarios
```

Results are written to `evaluation/eval_results.json` **and committed on
purpose** — they record the corpus path, retrieval config, judge model and
per-question outcomes alongside the scores, so a reader can audit the claim
rather than take it on trust.

### Optional retrieval upgrade

A cross-encoder reranker is wired in but **off by default** — it should only be
switched on once the harness shows it helps:

```bash
USE_RERANKER=1 python evaluation/ragas_eval.py
```

Change one variable at a time and re-run the harness as a gate.

---

## Known limitations

- **Answers are only as current as the documents.** The corpus mixes official
  PDFs with reference notes compiled from public web sources in July 2026. Fees,
  deadlines and rankings change every cycle — verify anything with financial or
  admission consequences against the institution directly.
- **The free Groq tier is the throughput ceiling** (~30 requests/minute, shared
  across everyone using a given deployment). Under load, answers are slower and
  can hit the rate-limit state.
- **No fine-tuned model is deployed.** See the honesty contract above.
- **Tabular PDFs are chunked as prose.** Fee tables can be split mid-row by the
  800-character splitter. Table-aware extraction is the next retrieval
  improvement worth measuring.
- **The corpus is trusted.** The prompt has no structural separation between
  instructions and retrieved text, so do not add unreviewed third-party
  documents to `data/`.
- **No conversation persistence.** Refreshing the page starts a new session.

---

## Configuration

All optional, via `.env` or Streamlit secrets:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Free at console.groq.com |
| `FAISS_PATH` | `./faiss_store` | Index location |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Serving model |
| `JUDGE_MODEL` | `llama-3.3-70b-versatile` | RAGAS judge |
| `RETRIEVAL_K` | `6` | Passages passed to the model |
| `RETRIEVAL_FETCH_K` | `30` | MMR candidate pool |
| `USE_RERANKER` | off | Cross-encoder reranking |
| `LOG_LEVEL` | `INFO` | stdout logging |
| `HF_TOKEN`, `PHI3_ENDPOINT_URL`, `MISTRAL_ENDPOINT_URL` | — | Fine-tuned backends |

---

## Deploying to Streamlit Community Cloud

1. Commit `faiss_store/` (the index must ship — there is no GPU to rebuild it).
2. Point a new Streamlit Cloud app at `app.py`.
3. Under **Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key"
   FAISS_PATH   = "./faiss_store"
   ```
4. Note that free Streamlit apps sleep after inactivity, so a cold link shows a
   wake-up screen first.

Never commit `.env` or `.streamlit/secrets.toml`.

---

## Licence

MIT — see [LICENSE](LICENSE). Note that the documents under `data/` are
third-party institutional material and are **not** covered by that licence;
verify redistribution rights before publishing the repository.
