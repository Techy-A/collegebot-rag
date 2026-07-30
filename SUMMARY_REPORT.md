# CollegeBot - build report

**Date:** 2026-07-30 · macOS · Python 3.9.6 · Groq free tier
**Scope:** all phases of the LLM-council plan ([council-improvements.md](council-improvements.md)),
plus wider Thapar data and a premium black interface.

---

## 1. What changed, in one table

| Area | Before | After |
|---|---|---|
| Session isolation | Conversation memory built inside `@st.cache_resource` - a process-global singleton, so concurrent visitors shared one mutable buffer | Only the stateless index is cached; memory lives in `st.session_state` |
| Model selector | 3 options; 2 silently fell back to Groq while the badge claimed "Phi-3-mini (fine-tuned)" | Only backends that can actually be served are listed; `get_llm()` raises rather than substituting |
| Response delivery | `streaming=False` - a spinner, then the whole answer | Token-by-token via `st.write_stream` |
| Chat rendering | Raw f-string interpolation into `unsafe_allow_html` divs (XSS surface; markdown bullets broke) | Native `st.chat_message` + `st.markdown()`, no HTML injection path |
| Citations | Bare filename chips | Numbered, expandable cards with human-readable title, **page number**, source year and the retrieved snippet |
| Refusals | Styled identically to answers | First-class amber "not in the knowledge base" state |
| Errors | Raw exception text persisted into the transcript | Typed handling; rate limits get a friendly message + backoff retry |
| "ONLINE" badge | Hardcoded literal | Bound to a real health check (index loaded **and** a backend configured) |
| Retrieval | MMR k=6, dense only | Hybrid dense + BM25 fused by Reciprocal Rank Fusion, k=8 |
| Corpus | 16 PDFs, 2153 chunks, including a **synthetic fixture** polluting real answers | 24 files, 2185 chunks, +8 researched Thapar documents, fixture removed |
| Evaluation | 12 questions written against a synthetic fixture; committed results were all-zero from a crash | 39-item gold set against the real corpus + refusal scoring + multi-turn |
| Judge model | Same 8B model graded its own output | `llama-3.3-70b-versatile` |
| Tests / CI / lint | None | 51 offline tests, GitHub Actions, ruff clean |
| Licence | README promised MIT; no file existed | `LICENSE` added, with a note on third-party documents |
| Repo hygiene | 15 MB dead `chroma_db/` incl. a stray duplicate `app.py`; duplicate `evaluation/ingest.py` | Removed |

---

## 2. Wider Thapar data

The corpus previously held 16 institutional PDFs and one **synthetic** handbook whose
invented facts (`erp.college.edu`, a fictional Rs 85,000 fee, fake library hours) were
being served as real answers. Eight researched documents were added, covering:

| Document | Contents |
|---|---|
| `thapar_overview_and_rankings.txt` | Founded 1956, deemed-to-be university, NAAC A++, NIRF 2025 (Engineering 29, Research 42, Overall 44), 250-acre campus, Dera Bassi second campus |
| `thapar_academics_and_programs.txt` | 7 Schools / 10 Departments / 13 Centres, B.Tech branches, M.Tech / M.Sc / MBA / MCA / PhD, LM Thapar School of Management |
| `thapar_admissions.txt` | 60% PCM eligibility (55% SC/ST), the two 50% admission channels, JEE Main route, counselling, helpline |
| `thapar_fees_and_scholarships.txt` | B.Tech tuition Rs 17,24,000, Rs 2,15,500/semester, hostel Rs 52,000–1,37,000, mess Rs 40–45k, 20–100% waivers |
| `thapar_hostel_and_campus.txt` | 16 hostels (10 boys / 6 girls), 10,000+ students, 24×7×365 library, 1 lakh+ books, sports, gym, health centre |
| `thapar_placements.txt` | 2025 average Rs 11.38 LPA (CSE Rs 16.50), 2024 highest Rs 1.23 crore, named recruiters |
| `thapar_research_and_innovation.txt` | 250+ patents, 1,500 papers/year, 15 centres, STEP / Venture Lab / Thapar Innovate |
| `thapar_student_life_and_contact.txt` | Postal address, helpline, Saturnalia (50th edition, Nov 2025), named societies |

Every document ends with the source URLs it was built from. Figures are labelled with
their year, because fees and rankings change annually.

The synthetic fixture was moved to `_fixtures/` - out of the index, still regenerable
via `ingest.py --sample` for tests. **This single change fixed a live wrong answer:**
"What are the library opening hours?" previously returned the fixture's invented
"Monday to Saturday, 8:00 AM to 8:00 PM"; it now correctly returns "open 24 hours a
day, on all 365 days of the year".

---

## 3. The interface

Kept black, as asked - but rebuilt for depth rather than outlines.

- **Layered near-blacks** (`#08080A` canvas over `#0E0E12` / `#15151B` / `#1D1D24`
  surfaces) with a single soft overhead glow, instead of boxes drawn in borders.
- **Hairline borders** at 6–16% white - structure you feel rather than see.
- **One metallic accent.** A champagne gradient carries the wordmark; indigo is
  reserved for interaction; emerald is reserved *exclusively* for "this is grounded"
  (the status dot and citation numbers). Restraint is what reads as premium.
- **Typography:** IBM Plex Sans / Mono, tabular figures, uppercase tracked mono
  micro-labels, generous line height.
- **Glass input bar** with a champagne focus ring.
- **Assistant turns** sit on a raised card; user turns stay on the canvas, so the eye
  lands on answers.
- **Accessibility fixed:** body text moved from `#52525B` (≈2.6:1 - failed WCAG AA) to
  `#8B8B94` (≈5.4:1 - passes). `prefers-reduced-motion` respected; mobile breakpoint added.
- Streamlit's default full-colour emoji avatars are desaturated to monochrome chips.

Everything lives in `assets/style.css` rather than a 100-line inline `<style>` block.

**The honesty rule the UI is built on:** it never asserts something the system did not
do. The status pill reflects a real check, the model chip names the model that actually
answered, a refusal looks like a refusal, and the heuristic scores are labelled as
keyword heuristics rather than dressed up as verified metrics.

---

## 4. Retrieval: measured, not assumed

Every configuration was scored on the same question: *does the key fact reach the
context at all?* (35 answerable gold questions, no LLM calls, so it is fast and free.)

| Configuration | Fact reaches context | Added latency |
|---|---|---|
| MMR k=6 fetch_k=30 (**previous default**) | 77.1% | - |
| similarity k=6 | 68.6% | - |
| BM25 only, k=8 | 80.0% | - |
| MMR k=8 fetch_k=60 | 80.0% | - |
| similarity(40) + cross-encoder → 6 | 85.7% | ~1.3 s/query |
| similarity(60) + cross-encoder → 8 | 88.6% | ~2.0 s/query |
| **hybrid dense + BM25 (RRF) → 8** | **88.6%** | **none** |
| hybrid + cross-encoder → 8 | 85.7% | ~1.4 s/query |
| hybrid k=16 | 91.4% | none (2× context tokens) |

**Hybrid retrieval ties the best reranker configuration at zero added latency**, and
stacking a reranker on top made it *worse*. Sparse retrieval earns its place because
this corpus is full of proper nouns, exact amounts and list headings ("TOP RECRUITERS",
"Rs 17,24,000", "A++") where lexical matching beats a 384-dim dense embedding.

k=8 was kept over k=16: the extra 2.8 points is not worth doubling context tokens on a
rate-limited free tier, and more distractors risk diluting faithfulness.

---

## 5. Evaluation results

Run against the **real** 2185-chunk corpus with the exact prompt and retrieval config
the app serves. Layer 1 is deterministic and needs no API calls.

| Round | Change | Refusal accuracy (≥0.95) | Fact recall (≥0.85) |
|---|---|---|---|
| 1 | Baseline: MMR k=6, new gold set | 0.9231 | 0.5714 |
| 2 | + hybrid dense+BM25 retrieval | 0.9744 | 0.7143 |
| 3 | + source year in context, exact-figure prompt rules | 0.9744 | 0.7714 |
| 4 | + archived the stale 2018/annual-report PDFs | 0.8718 | 0.7143 |
| 5 | **reverted to round 3 config** (what ships) | 0.8718 | 0.7429 |

### Read this table with the caveat it deserves

Rounds 3 and 5 ran **the same configuration** and scored 0.9744 vs 0.8718 on refusal
accuracy. Nothing changed between them but sampling: the harness was running at
temperature 0.05, and that run-to-run spread (~0.10) is **larger than most of the
improvements being measured**. Some of what rounds 2→3 appeared to show was noise, not
signal.

What survives that scrutiny:

- **The hybrid retrieval gain is real.** Fact recall 0.571 → 0.714 is a large step, and
  it is corroborated by the independent zero-API retrieval probe in §4, which is
  deterministic (77.1% → 88.6% on identical inputs).
- **The prompt/year-in-context gain (+5.7) is within noise** and should be treated as
  unproven rather than banked.
- **Round 4's regression is probably real** in direction (two metrics moved down
  together) but its magnitude is unreliable. It was reverted.

The harness has since been changed to **temperature 0.0** so future comparisons are
reproducible. That fix landed after these runs, so the numbers above retain the noise -
saying so is more useful than presenting a clean-looking table that cannot be reproduced.

**Honest bottom line:** fact recall sits around 0.74–0.77 against a 0.85 target, refusal
accuracy around 0.87–0.97 against 0.95. Neither target is reliably met yet. The
committed `evaluation/eval_results.json` is the round-5 run.

### What the remaining failures actually are

They are genuine, not measurement artefacts. All eight share one cause: the corpus
contains **two competing sources**, and the large stale ones sometimes outrank the
current curated ones.

- *"What NAAC grade?"* → "A, as per the Cycle 3 Reassessment in 2016" (from the 2018
  self-study report) instead of A++.
- *"NIRF ranking?"* → "20th … in NIRF 2018" instead of 29 in NIRF 2025.
- *"Which companies recruit?"* → e-waste collection vendors named in the annual report.

Surfacing each passage's year to the model and adding explicit exact-figure and recency
rules recovered part of this (+5.7 points in round 3). Fully closing it needs
per-document authority weighting at retrieval time - down-ranking a 2018 self-study
report when a 2025 source covers the same fact. That is the next measured change, and it
is deliberately left un-guessed rather than shipped untested.

---

## 6. Engineering

- **51 offline tests** - no API key, no vector store. They cover refusal detection,
  follow-up query construction, citation formatting (including 0-indexed → 1-indexed
  page numbers), prompt assembly, RRF fusion, the rate-limit retry path, ingest year
  tagging and digest dedup, and - most importantly - the factory's honesty contract:
  *a model must never be offered unless it can actually be served.*
- **CI** (`.github/workflows/ci.yml`): ruff lint + format check + pytest on every push.
- **Lint**: `ruff check` clean, `ruff format` applied across 15 files.
- **Structured logging** to stdout (query latency, passage count, refusals, rate-limit
  retries) - this is what made the false-refusal pattern visible.
- **`faiss_store/manifest.json`** records build time, embedding model, chunk settings and
  per-file effective years, so index staleness is observable rather than silent.

### Two bugs found and fixed during the work

1. **Streamlit rejects non-emoji avatars.** `st.chat_message(avatar="◆")` raises
   `StreamlitAPIException`. Switched to Streamlit's built-in icons, restyled in CSS.
2. **`st.rerun()` before the state append.** In the refusal path the record was appended
   *after* `st.rerun()`, which raises immediately - so refusals would have vanished from
   the transcript. The append now happens first.

---

## 7. Honest limitations

- **Neither evaluation target is reliably met.** Fact recall lands around 0.74–0.77
  against 0.85; refusal accuracy swings 0.87–0.97 against 0.95. Stated plainly rather
  than by quietly lowering the targets or quoting the best run.
- **Two out-of-scope questions were answered when they should have been declined** in the
  final run ("What is the B.Tech fee at IIT Delhi?", "What is my personal exam roll
  number?"). Earlier runs declined both. This is exactly the failure mode the refusal
  metric exists to catch, and it is not fixed.
- **The free Groq tier is the binding constraint** - roughly 30 requests/minute, shared
  across all users of a deployment. A full 39-question evaluation takes ~10 minutes at
  ~13–18 s per question, and RAGAS (layer 2) is opt-in for that reason.
- **RAGAS layer 2 was not run in this session.** Layer 1 (deterministic) is the reported
  evidence. Running RAGAS needs API budget the free tier cannot supply quickly.
- **No fine-tuned model is deployed.** The QLoRA notebook is real; no adapters exist, so
  those options do not appear in the UI. That is the honesty contract working.
- **Tabular PDFs are still chunked as prose.** Fee tables can be split mid-row. Table-aware
  extraction is the next retrieval improvement worth measuring.
- **The corpus is trusted.** There is no structural separation between instructions and
  retrieved text, so unreviewed third-party documents should not be added to `data/`.
- **Sub-agents were unavailable** for part of this work (the org monthly spend limit was
  reached), so the parallel research and review passes were done in the main session.

---

## 8. The LoRA fine-tuning track - considered, deliberately not completed

This repository sits in a folder called `LORA-PIPELINE`, so the omission deserves an
explicit account rather than silence.

### Why it was not trained

1. **No GPU on this machine.** `torch.cuda.is_available()` is `False` (macOS x86_64).
   QLoRA on Phi-3-mini/Mistral-7B needs a CUDA device; the notebook targets a Colab T4.
2. **It needs the owner's accounts.** Training requires a Google account for Colab and a
   *write-scope* Hugging Face token; serving requires an HF Inference Endpoint under that
   account. Those are not mine to create.
3. **The council advised against it as a quality lever, unanimously.** For narrow,
   closed-domain factual QA, a hardened grounding prompt over good retrieval is already
   near the ceiling; fine-tuning buys style and format adherence, not factual grounding.
4. **It would not fix the measured failure.** The gap in §5 is retrieval and source
   authority - the model quoting a 2018 self-study report over a 2025 source. No amount of
   fine-tuning corrects a stale passage in the context window.

### What was done for that track instead

- **The dishonest part was removed.** `llm_factory.available_models()` now lists a
  fine-tuned model only when its endpoint URL and HF token both exist, and `get_llm()`
  raises rather than silently serving Groq under a "Phi-3-mini (fine-tuned)" badge. Four
  tests enforce this contract.
- **The serving path is wired and ready.** Deploy an endpoint, set `PHI3_ENDPOINT_URL`,
  and the option appears in the UI with no code change.
- **The training dataset was generated** - `dataset/train.jsonl` (260 records) and
  `dataset/eval.jsonl` (29), built from the real corpus. It had never been generated
  before, so the notebook previously had nothing to consume.
- **The notebook was audited and six real defects fixed** (it had never been run):

  | Defect | Consequence | Fix |
  |---|---|---|
  | Installed `chromadb` | Minutes of Colab time and a build risk for a dependency the project dropped for FAISS | Removed, along with the leftover `/content/chroma_db` |
  | `unsloth@git+main`, unpinned `trl`/`peft`/`accelerate`/`bitsandbytes` | The most common way these notebooks break - Unsloth patches transformers in place and main drifts | Pinned |
  | `HF_TOKEN = "hf_your_token_here"` and the Groq key as cell literals | Tokens get committed and shared with the notebook | Read from Colab Secrets via `userdata.get()` |
  | `from langchain.text_splitter import ...` | Deprecated path under langchain 0.2.x | `langchain_text_splitters` |
  | Only pushed ~7.5 GB merged weights | A session timeout or full disk during the merge loses the entire training run | Adapter (a few MB) pushed first; merge is now an explicit `PUSH_MERGED` flag |
  | `save_pretrained_merged` ran unconditionally under an `if` | Would have executed even with merging disabled | Correctly indented into the branch |

### The dataset problem, and how it was fixed

Both existing generators produce data that would make the model worse:

- `--mode local` (keyword heuristics): **36% of answers under 60 characters, 39% of
  questions under 30, 16% dangling mid-phrase** - pairs like *"What are the marks?"* →
  *"60% (55% for SC/ST)marks in aggregate and Physics as one of the subject at..."*.
- `--mode groq` asks llama-3.1-8b to write the training data for a model of the same
  class, so the ceiling is its own output quality - and on the free tier it needs ~9s per
  call to stay inside 6,000 tokens/minute.

**The deeper flaw was the format, not the wording.** Both generators emit `input: ""` -
question in, answer out, no context. That trains the model to answer college questions
from parametric memory, which is the exact behaviour the grounding prompt exists to
prevent. Worse, the notebook's `format_alpaca` zipped only `instruction` and `output`, so
it **dropped the context field even if the dataset had one**. A fine-tune built that way
would have actively fought the RAG design and produced confident wrong answers.

So the dataset was hand-authored instead (`curated_dataset.py`, `--mode curated`):

| Metric | `--mode local` | curated |
|---|---|---|
| Answers under 60 chars | 36% | **5%** |
| Truncated / dangling answers | 16% | **0%** |
| Records carrying grounding context | 0% | **100%** |
| Answers using the house bullet style | 0% | **27%** |
| Mean answer length | 64 chars | **163 chars** |
| Refusal examples | **0** | **15 (21%)** |

Two design points worth naming:

1. **Every record carries its grounding passage in `input`**, so training mirrors
   inference. The notebook's formatter was rewritten to render it (and to append EOS,
   without which the model never learns to stop).
2. **Refusals are training data.** Neither generator can produce one, because both derive
   questions from the passage that contains the answer - so neither teaches the single
   behaviour this product depends on. The curated set pairs reasonable questions with
   context that genuinely does not answer them: other institutions' fees, personal
   records, prompt-injection attempts, subjective judgements ("are the hostel rooms
   tidy?"), and future predictions.

**Honest caveat: 70 pairs is a small set.** It is enough to shape behaviour (format
adherence and refusal discipline) with LoRA r=16 over 2 epochs, but it is not enough to
claim a broad capability gain. Extending it means adding entries to the two lists in
`curated_dataset.py` - the structure and the quality bar are established.

**Still not trained.** No GPU here, and the run needs the owner's Colab and HF accounts.
Everything upstream of pressing "Run all" is now correct and verified.

## 9. Reproduce it

```bash
cd collegebot-rag
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo 'GROQ_API_KEY=gsk_your_key' > .env

python ingest.py                 # 2185 chunks from 24 files, ~55 s
streamlit run app.py             # http://localhost:8501
pytest tests/ -q                 # 51 tests, offline
python evaluation/ragas_eval.py  # grounding checks on the real corpus
```

Committed evidence: `evaluation/eval_results.json` (scores **plus** corpus path,
retrieval config, judge model and every per-question outcome) and
`faiss_store/manifest.json`.
