"""
evaluation/ragas_eval.py  --  CollegeBot evaluation harness
============================================================
Runs the gold set from evaluation/gold_set.py against the REAL indexed corpus,
using the exact retrieval configuration and prompt that app.py serves (both
imported from rag.py / prompts.py -- there is no second copy to drift).

Two scoring layers:

  Layer 1  GROUNDING CHECKS (no API calls, always runs)
           - refusal accuracy: does it decline out-of-scope questions, and
             does it *not* decline answerable ones?  A false refusal on an
             in-corpus question is a real failure and was previously invisible.
           - fact recall: do answers contain the key figures from the source?
           These are deterministic, cheap and CI-friendly.

  Layer 2  RAGAS (opt-in with --ragas, needs API budget)
           faithfulness / answer_relevancy / context_precision.

Why RAGAS is opt-in: the free Groq tier allows ~30 requests/minute, and RAGAS
issues on the order of a hundred judge sub-calls for a few dozen questions.
A full run is therefore slow and prone to rate-limit timeouts that silently
become NaN -> 0.0 scores.  Layer 1 gives an honest signal on every run; Layer 2
is for when there is budget to spend.

Usage:
    python evaluation/ragas_eval.py                    # layer 1, whole gold set
    python evaluation/ragas_eval.py --limit 10         # layer 1, first 10 items
    python evaluation/ragas_eval.py --ragas --limit 8  # + RAGAS on a subset
    python evaluation/ragas_eval.py --multi-turn       # follow-up scenarios
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# RAGAS 0.1.x dispatches async judge calls across threads.  On Python 3.9 that
# raises "got Future attached to a different loop" and every metric collapses
# to 0.0.  nest_asyncio makes the loop re-entrant so the futures share one loop.
import nest_asyncio  # noqa: E402

import rag  # noqa: E402
from evaluation.gold_set import GOLD, MULTI_TURN  # noqa: E402
from llm_factory import GROQ_CHOICE, get_llm  # noqa: E402
from prompts import is_refusal  # noqa: E402

nest_asyncio.apply()

TARGETS = {
    "faithfulness": 0.92,
    "answer_relevancy": 0.87,
    "context_precision": 0.91,
    "refusal_accuracy": 0.95,
    "fact_recall": 0.85,
}

# The judge should be at least as capable as the system under test.  Using the
# same 8B model to grade its own output invites self-preference bias, and claim
# decomposition is exactly what small models are weakest at.  Evaluation is
# offline, so it can afford a slower, stronger model.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")


# ------------------------------------------------------------------
# Collection
# ------------------------------------------------------------------
def collect(store, llm, items: list) -> list:
    records = []
    n = len(items)
    print(f"\n{'-' * 66}\n  Running {n} gold questions against the real corpus\n{'-' * 66}")

    for i, item in enumerate(items, 1):
        q = item["question"]
        print(f"  [{i:>2}/{n}] ({item['domain']}) {q[:52]}")
        t0 = time.time()
        try:
            answer, docs = rag.answer(llm, q, store)
            err = None
        except Exception as exc:  # noqa: BLE001
            print(f"          ERROR: {exc}")
            answer, docs, err = "", [], str(exc)

        records.append(
            {
                "question": q,
                "answer": answer,
                "contexts": [d.page_content for d in docs],
                "ground_truth": item["ground_truth"],
                "domain": item["domain"],
                "expect_refusal": bool(item.get("expect_refusal")),
                "must_include": item.get("must_include", []),
                "refused": is_refusal(answer),
                "latency_s": round(time.time() - t0, 3),
                "error": err,
            }
        )
    return records


# ------------------------------------------------------------------
# Layer 1: grounding checks (no API)
# ------------------------------------------------------------------
def grounding_scores(records: list) -> dict:
    """
    refusal_accuracy -- refused exactly when it should have.
    fact_recall      -- answerable items whose answer contains the key figures.
    """
    refusal_ok = sum(1 for r in records if r["refused"] == r["expect_refusal"])

    checked, hits = 0, 0
    for r in records:
        if r["expect_refusal"] or not r["must_include"]:
            continue
        checked += 1
        low = r["answer"].lower()
        if all(token.lower() in low for token in r["must_include"]):
            hits += 1

    return {
        "refusal_accuracy": refusal_ok / len(records) if records else 0.0,
        "fact_recall": hits / checked if checked else 0.0,
        "n_records": len(records),
        "n_fact_checked": checked,
        "false_refusals": [
            r["question"] for r in records if r["refused"] and not r["expect_refusal"]
        ],
        "missed_refusals": [
            r["question"] for r in records if not r["refused"] and r["expect_refusal"]
        ],
        "fact_misses": [
            r["question"]
            for r in records
            if not r["expect_refusal"]
            and r["must_include"]
            and not all(t.lower() in r["answer"].lower() for t in r["must_include"])
        ],
    }


def per_domain(records: list) -> pd.DataFrame:
    rows = []
    for domain in sorted({r["domain"] for r in records}):
        subset = [r for r in records if r["domain"] == domain]
        ok = sum(1 for r in subset if r["refused"] == r["expect_refusal"])
        rows.append(
            {
                "domain": domain,
                "n": len(subset),
                "refusal_ok": f"{ok}/{len(subset)}",
                "avg_latency_s": round(sum(r["latency_s"] for r in subset) / len(subset), 2),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Layer 2: RAGAS (opt-in)
# ------------------------------------------------------------------
def compute_ragas(records: list) -> dict:
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    from ragas.run_config import RunConfig

    judge = LangchainLLMWrapper(
        ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=JUDGE_MODEL,
            temperature=0.0,
            timeout=300,
            max_retries=6,
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=rag.EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
        )
    )

    metrics = [faithfulness, answer_relevancy, context_precision]
    for m in metrics:
        m.llm = judge
        if hasattr(m, "embeddings"):
            m.embeddings = judge_emb

    # Refusals are excluded: "I don't have that information" is correct
    # behaviour but scores near zero on faithfulness and relevance, which would
    # drag the averages down and mask real generation quality.  Refusal
    # correctness is measured properly in layer 1.
    scored = [r for r in records if not r["refused"] and r["answer"]]
    if not scored:
        print("  No answerable records to score with RAGAS.")
        return {}

    print(f"\n  RAGAS judging {len(scored)} records with judge={JUDGE_MODEL}")
    print("  (free-tier rate limits make this slow; ~30-60s per record)")

    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in scored
        ]
    )
    try:
        # max_workers=4 stays under the free rate limit; 1 deadlocks the nested
        # loop on Python 3.9 and the RAGAS default of 16 trips 429s.
        scores = evaluate(dataset, metrics=metrics, run_config=RunConfig(max_workers=4))
        return {k: (0.0 if v != v else float(v)) for k, v in dict(scores).items()}
    except Exception as exc:  # noqa: BLE001
        print(f"\n  RAGAS failed: {exc}")
        return {}


# ------------------------------------------------------------------
# Multi-turn
# ------------------------------------------------------------------
def run_multi_turn(store, llm) -> list:
    print(f"\n{'-' * 66}\n  Multi-turn follow-up scenarios\n{'-' * 66}")
    results = []
    for scenario in MULTI_TURN:
        history, passed = [], True
        for turn in scenario["turns"]:
            answer, _docs = rag.answer(llm, turn["question"], store, history)
            low = answer.lower()
            ok = all(t.lower() in low for t in turn["must_include"])
            passed = passed and ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {scenario['name']}: {turn['question'][:44]}")
            history.append({"role": "user", "content": turn["question"]})
            history.append({"role": "assistant", "content": answer})
        results.append({"name": scenario["name"], "passed": passed})
    return results


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------
def report(grounding: dict, ragas: dict, domains: pd.DataFrame) -> bool:
    sep = "=" * 66
    print(f"\n{sep}\n  CollegeBot -- Evaluation Report (real corpus)\n{sep}")

    all_pass = True
    print("\n  GROUNDING CHECKS (deterministic, no API)")
    for key, label in (
        ("refusal_accuracy", "Refusal accuracy"),
        ("fact_recall", "Fact recall"),
    ):
        val = grounding.get(key, 0.0)
        target = TARGETS[key]
        passed = val >= target
        all_pass = all_pass and passed
        bar = "#" * int(val * 20)
        print(
            f"  [{'PASS' if passed else 'FAIL'}]  {label:<20} {val:.4f} "
            f"(target >= {target:.2f})  [{bar:<20}]"
        )
    print(
        f"         {grounding['n_records']} questions, {grounding['n_fact_checked']} fact-checked"
    )

    if ragas:
        print("\n  RAGAS (LLM-judged)")
        for key, label in (
            ("faithfulness", "Faithfulness"),
            ("answer_relevancy", "Answer relevance"),
            ("context_precision", "Context precision"),
        ):
            val = ragas.get(key, 0.0)
            target = TARGETS[key]
            passed = val >= target
            all_pass = all_pass and passed
            bar = "#" * int(val * 20)
            print(
                f"  [{'PASS' if passed else 'FAIL'}]  {label:<20} {val:.4f} "
                f"(target >= {target:.2f})  [{bar:<20}]"
            )

    if grounding["false_refusals"]:
        print("\n  FALSE REFUSALS (answer was in the corpus but it declined):")
        for q in grounding["false_refusals"]:
            print(f"    - {q}")
    if grounding["missed_refusals"]:
        print("\n  MISSED REFUSALS (should have declined but answered):")
        for q in grounding["missed_refusals"]:
            print(f"    - {q}")
    if grounding["fact_misses"]:
        print("\n  FACT MISSES (answer omitted the key figure):")
        for q in grounding["fact_misses"]:
            print(f"    - {q}")

    print(f"\n  PER-DOMAIN\n{domains.to_string(index=False)}")
    print(f"\n{sep}")
    print("  ALL TARGETS MET." if all_pass else "  ONE OR MORE TARGETS MISSED.")
    print(f"{sep}\n")
    return all_pass


def save(records: list, grounding: dict, ragas: dict) -> None:
    out_dir = PROJECT_ROOT / "evaluation"
    payload = {
        "corpus": os.getenv("FAISS_PATH", "./faiss_store"),
        "retrieval": {
            "k": rag.RETRIEVAL_K,
            "fetch_k": rag.RETRIEVAL_FETCH_K,
            "lambda_mult": rag.RETRIEVAL_LAMBDA,
            "hybrid_bm25": rag.USE_HYBRID,
            "reranker": rag.USE_RERANKER,
        },
        "judge_model": JUDGE_MODEL if ragas else None,
        "grounding": dict(grounding),
        "ragas": ragas,
        "targets": TARGETS,
        "results": [{k: v for k, v in r.items() if k != "contexts"} for r in records],
    }
    (out_dir / "eval_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(records).drop(columns=["contexts"]).to_csv(
        out_dir / "eval_results.csv", index=False
    )
    print(f"  Saved: {out_dir / 'eval_results.json'}")
    print(f"         {out_dir / 'eval_results.csv'}")


# ------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CollegeBot.")
    parser.add_argument("--ragas", action="store_true", help="Also run RAGAS (slow).")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N items.")
    parser.add_argument("--multi-turn", action="store_true", help="Follow-up scenarios.")
    parser.add_argument("--faiss", default=None, help="Override FAISS store path.")
    args = parser.parse_args()

    rag.configure_logging()
    print("\nCollegeBot Evaluation")
    print("=" * 66)

    store = rag.load_corpus(args.faiss)
    print(f"  Corpus: {store.ntotal} vectors")
    # temperature=0.0: the harness must be reproducible.  At 0.05 the same
    # configuration scored 0.974 and 0.872 refusal accuracy on two runs, which is
    # larger than most of the improvements being measured -- sampling noise was
    # masquerading as signal.
    llm = get_llm(GROQ_CHOICE, temperature=0.0, streaming=False)

    items = GOLD[: args.limit] if args.limit else GOLD
    records = collect(store, llm, items)

    grounding = grounding_scores(records)
    ragas = compute_ragas(records) if args.ragas else {}

    if args.multi_turn:
        mt = run_multi_turn(store, llm)
        print(f"\n  Multi-turn: {sum(1 for m in mt if m['passed'])}/{len(mt)} scenarios passed")

    ok = report(grounding, ragas, per_domain(records))
    save(records, grounding, ragas)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
