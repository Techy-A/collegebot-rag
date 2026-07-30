# LLM Council — "How should I improve CollegeBot?"

**Question:** What should I do to improve this project (CollegeBot, a RAG chatbot)? Make the web interface look great, improve overall project quality, and — what else is needed to make it genuinely great?

**Method:** four independent reviews, each entering the codebase through a different lens (first-principles, empirical/precedent, red-team, pragmatist) → anonymized peer review and ranking by four fresh reviewers → a single synthesis. Every reviewer read the real codebase and verified claims against it before answering.

## Aggregate ranking (unanimous — all 4 reviewers produced D > C > B > A)

| Rank | Member | Lens | Avg position |
|------|--------|------|--------------|
| 1 | Member 4 | Pragmatist | 1.0 |
| 2 | Member 3 | Red-team / skeptic | 2.0 |
| 3 | Member 2 | Empirical / precedent | 3.0 |
| 4 | Member 1 | First-principles | 4.0 |

**Where the council agreed (high confidence — all four, code-verified):** substance/honesty before polish; the cross-session `st.cache_resource` memory bug; the fake fine-tuned-model selector is an integrity risk; `unsafe_allow_html` chat bubbles = XSS + broken markdown bullets; `streaming=False` wastes Groq; the RAGAS numbers don't measure the shipped system; migrate to `st.chat_message`; page-numbered citations from already-captured metadata; rebuild the eval on the real corpus.

**Where they split:** the pragmatist (winner) uniquely isolated the shared-mutable-memory *root cause* and time-boxed everything; the empiricist uniquely caught the same-8B-model-as-its-own-judge flaw; the skeptic did the deepest eval forensics (two stores, crashed committed JSON, exact false refusal, ONLINE badge, Python 3.9 EOL, missing LICENSE); the first-principles member contributed the value hierarchy + the structured-extraction-for-lookups product bet but lost points for taking the eval numbers at face value. One unresolved factual tension (whether the committed all-zero `ragas_results.json` ran against the real store or the sample) is hedged: the committed eval evidence is unreliable either way and must be re-run.

---

# CHAIRMAN'S FINAL ANSWER

# CollegeBot: the council's verdict and action plan

## Core verdict

This is an above-average portfolio RAG project with a genuinely solid core — real MMR retrieval over 2,153 vectors from 16 real PDFs, a hardened grounding prompt with an explicit refusal clause, unusually thoughtful docstrings, and a two-layer eval discipline — and it does **not** need a rewrite. But for a product whose entire value proposition is "trustworthy answers grounded only in source PDFs," the currency is correctness and honesty, not aesthetics, and polishing the UI on top of unfixed substance actively backfires: slicker citation chips and green "faithfulness" pills launder unreliable evidence into something that *reads* as rigorous, and the technical reviewer most likely to judge this opens `llm_factory.py` before admiring the palette. So the resolution the whole council converged on is this: **fix trust, correctness, and a cross-session data-leak bug first; then make it beautiful — and recognize that the single highest-value form of "make it look great" here is disclosure (honest labels, real citations, honest uncertainty), which is a UI win and a substance win at the same time.** Where polish and honesty point the same way, do that first.

## The phased plan (the spine)

Work top to bottom. Effort is for a focused solo dev. Total for Phases 0–3 is roughly **6–9 focused days**; Phase 3 can run parallel to 2.

### Phase 0 — Stop the bleeding (~1 day)
- **Fix the cross-session shared-memory bug.** `load_pipeline` is `@st.cache_resource` and builds `ConversationBufferWindowMemory` *inside* the cached function. `st.cache_resource` is process-global, so concurrent visitors on Streamlit Cloud share one chain and **one mutable memory buffer — conversations bleed across users.** Correctness + privacy bug. Same root cause as the temperature-slider full-reload and the "Clear conversation" global cache wipe. Fix once: cache only stateless embeddings+FAISS+retriever keyed on `faiss_path`; put memory + chain in `st.session_state`; Clear resets `session_state` only. **~2–3 hrs. First.**
- **Remove/relabel the fake model selector.** `_build_phi3`/`_build_mistral` fail the CUDA check, `warnings.warn` to stderr, silently return Groq — while the badge asserts "Phi-3-mini (fine-tuned)." `requirements.txt` has no torch/transformers/bitsandbytes/peft, so that path never ran as committed. Delete the options (~15 min) or make fallback a visible `st.toast`; move the fine-tune story to the README.
- **Stop persisting raw exception text as a chat turn** (a 429 becomes a permanent bot bubble). Ephemeral `st.error`. **~30 min.**
- **Wire the "ONLINE" badge to real health** (currently an unconditional literal). **~30 min.**
- **Repo hygiene + licensing:** delete tracked `chroma_db/` (15.9 MB legacy sqlite + a stray duplicate `app.py`); add the `LICENSE` the README claims; schedule a Python 3.9→3.11/3.12 bump (3.9 is EOL). **~1 hr.**

### Phase 1 — Make the interface great, natively (~1.5–2 days)
- **Port to `st.chat_message` + `st.markdown(content)`** — kills ~140 lines of HTML, fixes markdown bullets, closes XSS, in one move. **~4–6 hrs.**
- **Turn on streaming** (keep the chain; stream just the final answer via `st.write_stream`). **~2–4 hrs.**
- **Real page-numbered citation cards** from `result["source_documents"]`. **~2–3 hrs.**
- **Distinct refusal state + honest metric labels + friendly error/rate-limit states.** **~2–3 hrs.**
- **Declutter + small batch:** Advanced expander for temperature; favicon; cold-start message; mobile width; contrast fix; affiliation disclaimer. **~2–3 hrs.**

### Phase 2 — Correctness and eval (~2–3 days)
- **Rebuild the eval against the real corpus** — 50–100 human-verified Q&A over the real `faiss_store`, stratified across six domains + out-of-scope refusals, exact `app.py` prompt+config, committed. **~1.5–2 days.**
- **Point the judge at a stronger model** (not the 8B judging itself). **~30 min.**
- **Retrieval upgrades, one variable at a time, gated by the harness:** cross-encoder reranker *or* `bge-small` swap first, measure, then BM25 hybrid + table extraction. **~1 day first lever.**
- **Investigate false-refusal-on-in-corpus pattern** (hostel/library misses ↔ 9–22 s latencies = rate-limit-retry tell). **~half day.**
- **Add multi-turn + adversarial eval cases; fix fee-year/hostel duplicate-PDF staleness.** **~half day.**

### Phase 3 — Engineering hygiene (~1.5–2 days, parallelizable)
- **pytest** (`get_llm` dispatch incl. fallback, ingestion smoke, golden-set regression on refusal rate + must-include facts; `quick_score.py` is pure functions) + **GitHub Actions** (ruff + pytest).
- **Structured logging to stdout;** **retry/backoff (tenacity)** catching Groq 429; **ruff + type hints; decompose `app.py`; document the `ragas==0.1.21` pin.** Optionally fold in the LCEL migration here.

### Phase 4 — Polish and portfolio (~0.5 day)
Feedback thumbs, conversation export, "How this works / Known Limitations" README, screenshots + 15–30 s GIF + hosted demo link, freshness stamp, privacy line.

## Make the interface great
The highest-leverage UI work is **disclosure, not decoration.** Keep the tasteful dark/IBM Plex identity. Then:
1. **`st.chat_message` + `st.markdown(content)`** — fixes XSS (raw `msg["content"]` in `unsafe_allow_html`, including retrieved-PDF content as an injection vector), fixes bullets (literal `\n`/`-` don't render in a raw HTML block), removes CSS combat. Re-point the theme at `[data-testid="stChatMessage"]`.
2. **Streaming** — no chain rewrite needed; call the retriever, build the prompt, `llm.stream(...)` into `st.write_stream`.
3. **Real citations** — `PyPDFLoader` already stamps `metadata["page"]` and chunk text into `result["source_documents"]`; you discard all but the filename. Numbered `[1][2]` on the claim → `st.expander` with title + page + snippet.
4. **Honest uncertainty as a first-class state** — distinct style for "I don't know"; label eval pills "automated heuristic self-check, not a guarantee."
5. **Declutter + finish** — Advanced expander for temperature; extend suggestion chips to all six domains; favicon; cold-start message; mobile width; fix contrast (`#52525B` on `#09090B` ≈ 2.57:1, fails AA); add "unofficial — verify all information" disclaimer.

## Improve overall quality
- **Rebuild the evaluation (the most important substance fix).** `EVAL_DATASET` ground truths match `create_sample_data()`'s synthetic fixture verbatim; the 0.826/0.869/1.0 came from a 5-Q run on a 6-vector sample store with a lighter prompt. **The committed `ragas_results.json` is unreliable either way (all zeros from a Py3.9 crash) and must be re-run against the real corpus with the production prompt/config.** Do not cite current numbers.
- **Fix the judge** — `ragas_eval.py` uses the same 8B as system + judge; use a 70B-class free Groq judge.
- **Retrieval, in payoff order, one variable at a time:** `bge-small-en-v1.5` swap → cross-encoder reranker (do this first, measure) → BM25 hybrid + RRF → table-aware extraction (you depend on `unstructured` but only use it for HTML; `CHUNK_SIZE=800` slices fee tables).
- **Faithfulness gap:** precision 1.0 but faithfulness 0.826 → generation grounding, not retrieval, is the bottleneck (~1 in 6 fails). Add a generate-then-verify step (claim+quote pairs, confirm quote in context).
- **Multi-turn is untested;** add follow-up + adversarial cases. **Data hygiene:** version/dedupe the two fee-year and two hostel PDFs (MMR blends cross-year); consider retiring the 17.6 MB annual report noise.
- **Tests/CI/error-handling** per Phase 3. `ConversationalRetrievalChain` runs a condense call before every follow-up (2 Groq calls/turn, halving throughput vs 30 req/min) — an eventual LCEL migration helps, but it's not a prerequisite for streaming.

## What else is needed
- **Feedback loop** (thumbs logged with Q+A+sources+latency) — turns static eval into a living one and sources the real queries your gold set lacks.
- **Observability** (structured stdout logging) — zero today; would have surfaced the false-refusal pattern. Not LangSmith/Helicone yet.
- **Freshness** — "sources current as of <date>", index manifest (timestamp/doc list/hashes), scripted rebuild.
- **The bigger product bet — structured extraction for lookups:** extract fee/date tables to structured records at ingestion, answer lookups directly, fall back to vector RAG for prose. "Usually right" → "reliably right" on the highest-stakes questions. Sequence *after* the eval harness exists.
- **Privacy stance** before logging (financial-aid/disability/grievance are sensitive).
- **Portfolio** (highest ROI/min once UI lands): only 4 commits reads as thin — add README screenshots, a 15–30 s GIF, a hosted demo link, architecture diagram, honest "Known Limitations" (explicit-boundary systems are trusted *more*). Plan for Streamlit Cloud sleep-after-inactivity. 10-min data-licensing check on the PDFs (esp. NAAC/annual report).

## Explicitly do NOT do
- **NOT** migrate to Next.js/React + FastAPI now (3–7 days for a payoff Phase 1 already reaches). Only if the goal becomes full-stack demonstration, or Streamlit genuinely can't do the UI.
- **NOT** finish QLoRA as a quality lever (1–2 days Colab for a model likely *worse* than well-prompted Groq on this narrow task). Keep it as an honest README benchmark.
- **NOT** stack reranker + BM25 + embedding-swap at once. One variable, measured.
- **NOT** chase a LangChain/RAGAS major upgrade now (pins exist for a reason; document them).
- **NOT** add hosted observability SaaS, auth/multi-tenancy, knowledge graph, multi-agent planning, a paid model, a hosted vector DB, or `mypy --strict` — resume-driven complexity reviewers discount.

**If you do only five things, in order:** (1) fix the cross-session memory/caching bug; (2) migrate to `st.chat_message`; (3) turn on streaming; (4) page-numbered citation cards; (5) rebuild the eval against the real corpus and commit real numbers.
