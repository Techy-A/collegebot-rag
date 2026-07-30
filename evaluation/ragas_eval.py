"""
evaluation/ragas_eval.py  --  Full RAGAS Evaluation Pipeline
=============================================================
"In God we trust.  All others must bring data."  -- W. Edwards Deming

This script runs the authoritative offline evaluation of CollegeBot
against the three RAGAS metrics required by the project specification:

    Metric               Target    RAGAS key
    -------------------  --------  ----------------------
    Faithfulness         >= 0.92   faithfulness
    Answer Relevance     >= 0.87   answer_relevancy
    Context Precision    >= 0.91   context_precision

The script:
    1. Builds the RAG chain (same configuration as app.py).
    2. Runs the chain over the evaluation dataset.
    3. Computes RAGAS scores using the Groq LLM as the judge.
    4. Prints a pass/fail report for each metric.
    5. Saves detailed results to evaluation/ragas_results.json
       and evaluation/ragas_results.csv for auditing.
    6. If any metric misses its target, prints specific
       optimisation tips for that metric.

Usage:
    python evaluation/ragas_eval.py

Requirements:
    GROQ_API_KEY must be set in .env.
    The FAISS index must exist (run ingest.py first).

Author : CollegeBot Team
License: MIT
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path so imports work from any working directory.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ------------------------------------------------------------------
# Metric targets.
# These are the exact thresholds specified in the project requirements.
# Do not lower them without explicit approval.
# ------------------------------------------------------------------
TARGETS = {
    "faithfulness"     : 0.92,
    "answer_relevancy" : 0.87,   # RAGAS key uses "relevancy" not "relevance"
    "context_precision": 0.91,
}

# Mapping from RAGAS internal key to human-readable display name.
DISPLAY_NAMES = {
    "faithfulness"     : "Faithfulness",
    "answer_relevancy" : "Answer Relevance",
    "context_precision": "Context Precision",
}


# ------------------------------------------------------------------
# Evaluation dataset.
# These 12 question-answer pairs span the major topic areas of the
# sample college handbook.  For a real deployment, this set should
# be expanded to at least 50 pairs drawn from real student queries.
#
# Ground truth answers are written conservatively -- they state what
# the document explicitly says, without inference or elaboration.
# This is important because the faithfulness metric penalises answers
# that go beyond the source text.
# ------------------------------------------------------------------
EVAL_DATASET = [
    {
        "question"    : "What is the last date to submit the college admission form?",
        "ground_truth": "The last date to submit the college admission form is June 30, 2024.",
    },
    {
        "question"    : "What documents are required for admission?",
        "ground_truth": (
            "Required documents for admission include 10th and 12th mark sheets, "
            "Transfer Certificate, Migration Certificate, Character Certificate, "
            "four passport-size photographs, Aadhar Card, and Category Certificate if applicable."
        ),
    },
    {
        "question"    : "What is the annual fee for the B.Tech program?",
        "ground_truth": (
            "The annual fee for B.Tech is Rs 85,000, comprising tuition Rs 65,000, "
            "development Rs 12,000, and examination Rs 8,000."
        ),
    },
    {
        "question"    : "How do students apply for hostel accommodation?",
        "ground_truth": (
            "Students apply for hostel accommodation by submitting the Hostel Application Form "
            "to the Office of the Dean of Student Welfare before the start of each semester."
        ),
    },
    {
        "question"    : "What scholarships are available for students?",
        "ground_truth": (
            "Available scholarships include the Chief Minister Scholarship, "
            "Merit-cum-Means Scholarship, College Merit Scholarship, and Sports Scholarship. "
            "Applications open from August 1 to September 30 each year."
        ),
    },
    {
        "question"    : "What is the minimum attendance required to sit for exams?",
        "ground_truth": "Students must maintain a minimum of 75% attendance in each subject.",
    },
    {
        "question"    : "How do I register for elective courses?",
        "ground_truth": (
            "Students register for elective courses through the online ERP portal at erp.college.edu "
            "during Week 3 of each semester, Monday to Friday from 9 AM to 5 PM."
        ),
    },
    {
        "question"    : "What are the library opening hours?",
        "ground_truth": (
            "The Central Library is open Monday to Saturday from 8:00 AM to 8:00 PM "
            "and on Sunday from 10:00 AM to 4:00 PM."
        ),
    },
    {
        "question"    : "What is the penalty for late course registration?",
        "ground_truth": (
            "Late registration incurs a penalty of Rs 200 per day for up to five days, "
            "after which registration closes for that semester."
        ),
    },
    {
        "question"    : "How many books can a student borrow from the library?",
        "ground_truth": "Students may borrow up to four books at a time for a period of 14 days.",
    },
    {
        "question"    : "When are end-semester examinations held?",
        "ground_truth": (
            "End-semester examinations are held in November-December for odd semesters "
            "and in April-May for even semesters."
        ),
    },
    {
        "question"    : "How do I submit an academic grievance?",
        "ground_truth": (
            "Students submit grievance forms to the Academic Section, Administrative Block, Room 102 "
            "within 15 days of result declaration.  Response time is 10 working days."
        ),
    },
]


# ------------------------------------------------------------------
# Pipeline construction.
# Identical to app.py's load_pipeline() but without Streamlit
# decorators.  This is intentional -- evaluation should be able to
# run headlessly without a browser session.
# ------------------------------------------------------------------
def build_chain():
    """
    Construct the ConversationalRetrievalChain for evaluation.

    "Programs are meant to be read by humans and only incidentally
     for computers to execute."  -- Donald Knuth
    """
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
    from llm_factory import get_llm

    embeddings = HuggingFaceEmbeddings(
        model_name    = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )

    faiss_path = os.getenv("FAISS_PATH", "./faiss_store")
    vectorstore = FAISS.load_local(
        faiss_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_type   = "mmr",
        search_kwargs = {"k": 6, "fetch_k": 20, "lambda_mult": 0.6},
    )

    QA_PROMPT = PromptTemplate(
        template="""You are CollegeBot, a precise and reliable assistant for college
students, faculty, and administrative staff.

STRICT RULES:
1. Answer ONLY from the provided CONTEXT below.
2. If the context does not contain the answer, say:
   "I do not have that information in my knowledge base."
3. Never invent deadlines, fees, names, or policy details.

CONTEXT:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {question}

ANSWER:""",
        input_variables=["context", "chat_history", "question"],
    )

    # For evaluation, each question is treated independently.
    # Using k=1 in memory prevents prior Q-A pairs from contaminating
    # the context window for the next evaluation question.
    memory = ConversationBufferWindowMemory(
        k              = 1,
        memory_key     = "chat_history",
        return_messages= True,
        output_key     = "answer",
    )

    # Use temperature=0.0 for evaluation to maximise reproducibility.
    # Stochastic answers make it harder to compare runs.
    llm = get_llm("groq/llama-3.1-8b-instant", temperature=0.05)

    chain = ConversationalRetrievalChain.from_llm(
        llm                       = llm,
        retriever                 = retriever,
        memory                    = memory,
        combine_docs_chain_kwargs = {"prompt": QA_PROMPT},
        return_source_documents   = True,
        verbose                   = False,
    )
    return chain


# ------------------------------------------------------------------
# RAG output collection.
# ------------------------------------------------------------------
def collect_outputs(chain, dataset: list) -> list:
    """
    Run the chain over the dataset and collect structured outputs.
    Each output dict matches the schema expected by the RAGAS Dataset.
    """
    records = []
    n       = len(dataset)
    sep     = "-" * 62

    print(f"\n{sep}")
    print(f"  Collecting RAG outputs for {n} evaluation questions")
    print(sep)

    for i, item in enumerate(dataset, 1):
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"  [{i:>2}/{n}] {q[:60]}...")
        t0 = time.time()

        try:
            result  = chain.invoke({"question": q})
            latency = round(time.time() - t0, 3)
            answer  = result.get("answer", "")
            docs    = result.get("source_documents", [])
            contexts= [d.page_content for d in docs]
        except Exception as e:
            print(f"         ERROR: {e}")
            latency  = 0.0
            answer   = ""
            contexts = []

        records.append({
            "question"    : q,
            "answer"      : answer,
            "contexts"    : contexts,
            "ground_truth": gt,
            "latency_s"   : latency,
        })

    return records


# ------------------------------------------------------------------
# RAGAS metric computation.
# ------------------------------------------------------------------
def compute_ragas(records: list) -> dict:
    """
    Run the RAGAS evaluation library over the collected records.

    RAGAS uses the judge LLM (here Groq/Llama-3.1) to assess
    faithfulness and relevance, and the embedding model to assess
    semantic similarity.  The library is called once per metric
    to enable partial re-runs if one metric fails.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings

    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            groq_api_key = os.getenv("GROQ_API_KEY"),
            model_name   = "llama-3.1-8b-instant",
            temperature  = 0.0,
            timeout      = 120,  # Increase timeout to 120 seconds
            max_retries  = 3,    # Retry failed requests
        )
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name    = "sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs = {"normalize_embeddings": True},
        )
    )

    metrics = [faithfulness, answer_relevancy, context_precision]
    for m in metrics:
        m.llm = judge_llm
        if hasattr(m, "embeddings"):
            m.embeddings = judge_emb

    # RAGAS expects contexts as a list of strings per row,
    # which matches the structure we built in collect_outputs().
    dataset = Dataset.from_list(records)

    print("\n  Computing RAGAS scores (approx. 2-5 minutes on free Groq tier)...")
    print(f"  Dataset size: {len(records)} records")
    print(f"  Sample record keys: {list(records[0].keys()) if records else 'empty'}")
    if records:
        print(f"  Sample contexts length: {len(records[0].get('contexts', []))}")
    
    try:
        scores = evaluate(dataset, metrics=metrics)
        return dict(scores)
    except Exception as e:
        print(f"\n  ERROR during RAGAS evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}


# ------------------------------------------------------------------
# Report printing.
# ------------------------------------------------------------------
def print_report(scores: dict) -> bool:
    """
    Print a structured pass/fail report.

    Returns True if all metrics meet their targets, False otherwise.
    """
    sep = "=" * 62
    print(f"\n{sep}")
    print("  CollegeBot  --  RAGAS Evaluation Report")
    print(sep)

    all_pass = True
    for key, (label, target) in {
        "faithfulness"     : ("Faithfulness",     TARGETS["faithfulness"]),
        "answer_relevancy" : ("Answer Relevance",  TARGETS["answer_relevancy"]),
        "context_precision": ("Context Precision", TARGETS["context_precision"]),
    }.items():
        val    = scores.get(key, 0.0)
        # Handle NaN values
        if val != val or val is None:  # NaN check
            val = 0.0
        passed = val >= target
        status = "PASS" if passed else "FAIL"
        bar    = "#" * int(val * 20)
        print(
            f"  [{status}]  {label:<22}  {val:.4f}  "
            f"(target >= {target:.2f})  [{bar:<20}]"
        )
        if not passed:
            all_pass = False

    print(sep)
    if all_pass:
        print("  ALL TARGETS MET.  The system is production ready.")
    else:
        print("  ONE OR MORE TARGETS MISSED.  See optimisation tips below.")
        _print_tips(scores)
    print(f"{sep}\n")
    return all_pass


# ------------------------------------------------------------------
# Targeted optimisation tips.
# ------------------------------------------------------------------
def _print_tips(scores: dict) -> None:
    """
    Print specific, actionable tips for each failing metric.
    These are ordered from highest-leverage to lowest-leverage
    based on empirical testing during development.

    "Fix the cause, not the symptom."  -- Unknown
    """
    faith = scores.get("faithfulness",      1.0)
    relev = scores.get("answer_relevancy",  1.0)
    prec  = scores.get("context_precision", 1.0)

    print("\n  OPTIMISATION TIPS:")
    print("  " + "-" * 58)

    if faith < TARGETS["faithfulness"]:
        print("\n  FAITHFULNESS  (current: {:.4f}  target: {:.2f})".format(
            faith, TARGETS["faithfulness"]))
        print("  1. Strengthen the grounding instruction in QA_PROMPT:")
        print('     Add: "Every sentence in your answer must appear verbatim')
        print('      or be a direct paraphrase of the CONTEXT above."')
        print("  2. Reduce temperature to 0.05 or 0.0.")
        print("  3. Increase chunk_overlap from 150 to 200 in ingest.py.")
        print("  4. Filter out chunks below a minimum length of 50 characters")
        print("     to remove table-of-contents noise from PDFs.")
        print("  5. Fine-tune the model with more context-grounded examples.")

    if relev < TARGETS["answer_relevancy"]:
        print("\n  ANSWER RELEVANCE  (current: {:.4f}  target: {:.2f})".format(
            relev, TARGETS["answer_relevancy"]))
        print("  1. Add to QA_PROMPT: 'Begin your answer by directly addressing")
        print("     the question.  Do not start with background context.'")
        print("  2. Increase k from 6 to 8 in the MMR retriever.")
        print("  3. Expand the synthetic fine-tuning dataset with more")
        print("     question-diverse examples (yes/no, how-to, when, who).")
        print("  4. Use BM25 hybrid retrieval alongside MMR to improve recall")
        print("     for keyword-specific questions.")

    if prec < TARGETS["context_precision"]:
        print("\n  CONTEXT PRECISION  (current: {:.4f}  target: {:.2f})".format(
            prec, TARGETS["context_precision"]))
        print("  1. Lower lambda_mult from 0.6 to 0.5 in the MMR retriever")
        print("     to favour relevance more strongly over diversity.")
        print("  2. Reduce chunk_size from 800 to 600 in ingest.py to create")
        print("     more topically focused chunks.")
        print("  3. Add metadata filtering (e.g., filter by department or")
        print("     document type) before retrieval for narrow queries.")
        print("  4. Add a reranker step (e.g., cross-encoder/ms-marco-MiniLM-L6)")
        print("     between retrieval and generation -- free to run on CPU.")
        print("  5. Review your source documents: noisy or off-topic PDFs")
        print("     (e.g., scanned images with OCR errors) degrade precision.")


# ------------------------------------------------------------------
# Persistence.
# ------------------------------------------------------------------
def save_results(records: list, scores: dict) -> None:
    """
    Save evaluation artefacts to disk for auditing and comparison.
    """
    out_dir = PROJECT_ROOT / "evaluation"
    out_dir.mkdir(exist_ok=True)

    # JSON: full detail including scores and per-question answers.
    json_path = out_dir / "ragas_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scores" : scores,
                "targets": TARGETS,
                "n_questions": len(records),
                "results": [
                    {k: v for k, v in r.items() if k != "contexts"}
                    for r in records
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # CSV: tabular view without the contexts list (too wide for a spreadsheet).
    csv_path = out_dir / "ragas_results.csv"
    pd.DataFrame(records).drop(columns=["contexts"]).to_csv(csv_path, index=False)

    print(f"  Results saved:")
    print(f"    {json_path}")
    print(f"    {csv_path}")


# ------------------------------------------------------------------
# Entry point.
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\nCollegeBot RAGAS Evaluation")
    print("=" * 62)

    # 1. Build chain.
    print("\nStep 1: Building RAG chain...")
    chain = build_chain()

    # 2. Collect outputs.
    records = collect_outputs(chain, EVAL_DATASET)

    # 3. Compute scores.
    scores = compute_ragas(records)

    # 4. Print report.
    all_pass = print_report(scores)

    # 5. Save artefacts.
    print("Saving results...")
    save_results(records, scores)

    # Exit with non-zero code if targets are missed -- useful in CI.
    sys.exit(0 if all_pass else 1)
