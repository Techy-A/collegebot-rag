"""
generate_dataset.py  --  Synthetic Q-A Dataset Generator for QLoRA Fine-Tuning
================================================================================
"Without data you are just another person with an opinion."
    -- W. Edwards Deming

This script generates a synthetic question-answer dataset from your
college documents.  The dataset is formatted in the Alpaca instruction
format required by the Unsloth fine-tuning notebook.

Two generation modes are supported:

    1. LOCAL (default)
       Uses rule-based extraction to create Q-A pairs from the
       document chunks.  No API calls.  Fast but lower quality.

    2. GROQ-ASSISTED
       Uses the Groq LLM to generate diverse, natural Q-A pairs
       from each chunk.  Requires GROQ_API_KEY.  Higher quality.
       Set --mode groq to activate.

Output format (Alpaca / ShareGPT):
    Each record is a dict with fields:
        instruction -- the question
        input       -- empty string (context injected separately)
        output      -- the answer

The final dataset is saved as:
    dataset/train.jsonl  -- 90% of records (for fine-tuning)
    dataset/eval.jsonl   -- 10% of records (for validation loss)

Usage:
    python generate_dataset.py --mode local
    python generate_dataset.py --mode groq --target 500

Author : CollegeBot Team
License: MIT
"""

import os
import re
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()


# ------------------------------------------------------------------
# Question templates.
# These templates are instantiated with named slots extracted from
# the chunk text.  They cover the major question archetypes found
# in real student queries: what, how, when, who, where, can I.
# ------------------------------------------------------------------
QUESTION_TEMPLATES = [
    # Factual retrieval
    "What is the {topic}?",
    "What are the {topic}?",
    "What is the policy regarding {topic}?",
    "What is the procedure for {topic}?",
    "How many {topic}?",
    "How much is the {topic}?",
    # Procedural
    "How do I {action}?",
    "How can a student {action}?",
    "What steps are required to {action}?",
    # Temporal
    "When is the deadline for {topic}?",
    "When does {topic} take place?",
    "What is the last date for {topic}?",
    # Locational
    "Where is the {topic} located?",
    "Where do I submit {topic}?",
    # Eligibility
    "Who is eligible for {topic}?",
    "What are the eligibility criteria for {topic}?",
    "Can a student {action} if they have less than 75% attendance?",
]


# ------------------------------------------------------------------
# Local (rule-based) dataset generation.
# ------------------------------------------------------------------
def generate_local(chunks: List[str], target: int) -> List[Dict]:
    """
    Generate Q-A pairs from chunk text using pattern matching.

    The heuristic works as follows:
      1. Each chunk is scanned for sentences containing key nouns
         (fee, deadline, requirement, scholarship, etc.).
      2. For each matched sentence, a template question is instantiated
         with the key noun as the topic slot.
      3. The matched sentence is used as the answer.

    This approach produces lower-quality training data than GPT/Groq
    generation, but it is entirely free and deterministic.

    "The best programs are those written for clarity of expression
     to human readers, not for speed of machine execution."
        -- Donald Knuth
    """
    key_nouns = [
        "fee", "fees", "scholarship", "admission", "attendance",
        "hostel", "library", "deadline", "examination", "registration",
        "document", "documents", "certificate", "grievance", "marks",
        "internship", "placement", "canteen", "sports", "club",
        "semester", "syllabus", "result", "revaluation", "condonation",
    ]

    records = []
    random.seed(42)

    for chunk in chunks:
        sentences = [s.strip() for s in re.split(r"[.!?\n]+", chunk) if len(s.strip()) > 30]
        for sent in sentences:
            lower = sent.lower()
            for noun in key_nouns:
                if noun in lower:
                    # Select a random question template.
                    # "can I" templates require an action; others use topic.
                    template = random.choice(QUESTION_TEMPLATES)
                    if "{action}" in template:
                        # Derive a gerund phrase from the sentence.
                        action = _extract_action(sent, noun)
                        question = template.format(action=action)
                    else:
                        question = template.format(topic=noun)

                    records.append({
                        "instruction": question,
                        "input"      : "",
                        "output"     : sent.strip(),
                    })
                    break   # one record per sentence is sufficient

        if len(records) >= target:
            break

    # Deduplicate on (instruction, output) pairs.
    seen    = set()
    unique  = []
    for r in records:
        key = (r["instruction"], r["output"][:60])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    random.shuffle(unique)
    return unique[:target]


def _extract_action(sentence: str, noun: str) -> str:
    """
    Heuristically derive an action phrase from a sentence containing a key noun.
    e.g. "Students must submit the fee by July 15." -> "submit the fee"
    """
    patterns = [
        r"must (.+?)(?:\s+by|\s+before|\s+to|\.|$)",
        r"should (.+?)(?:\s+by|\s+before|\s+to|\.|$)",
        r"can (.+?)(?:\s+by|\s+before|\s+to|\.|$)",
        r"required to (.+?)(?:\s+by|\s+before|\s+to|\.|$)",
        r"need to (.+?)(?:\s+by|\s+before|\s+to|\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, sentence, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return f"apply for {noun}"


# ------------------------------------------------------------------
# Groq-assisted dataset generation.
# ------------------------------------------------------------------
def generate_groq(chunks: List[str], target: int) -> List[Dict]:
    """
    Use the Groq LLM to generate natural, diverse Q-A pairs.
    Each chunk is sent to the model with a structured prompt that
    requests multiple questions and corresponding answers.

    Rate limit management:
        Groq free tier: 6,000 tokens/minute.
        This function sleeps 1 second between requests to stay well
        below the limit.

    "Measure twice, cut once."  -- Every carpenter who ever lived.
    """
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set.  Cannot use Groq-assisted generation.  "
            "Use --mode local instead, or set GROQ_API_KEY in .env."
        )

    client = Groq(api_key=api_key)
    records = []

    # How many questions to request per chunk.
    qs_per_chunk = 3

    print(f"\n  Requesting ~{qs_per_chunk} Q-A pairs per chunk from Groq LLM...")

    for i, chunk in enumerate(chunks):
        if len(records) >= target:
            break

        prompt = f"""You are an expert at generating training data for educational chatbots.
Given the following text from a college student handbook, generate exactly {qs_per_chunk} diverse
question-answer pairs.  Questions must be answerable from the text alone.

Rules:
- Each question must be different in type (what, how, when, where, who, can).
- Answers must be copied or paraphrased from the text -- no invented information.
- Format: each pair on two lines, starting with Q: and A:.  No numbering.

TEXT:
{chunk[:600]}

Generate {qs_per_chunk} Q-A pairs:"""

        try:
            response = client.chat.completions.create(
                model       = "llama-3.1-8b-instant",
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.7,
                max_tokens  = 600,
            )
            text = response.choices[0].message.content or ""
            pairs = _parse_qa_pairs(text)
            records.extend(pairs)
            print(f"  Chunk {i+1}/{len(chunks)}: extracted {len(pairs)} pairs ({len(records)} total)")
        except Exception as e:
            print(f"  Chunk {i+1}: Groq error -- {e}")

        time.sleep(1.0)   # Stay under the 6,000 tokens/minute free rate limit.

    random.shuffle(records)
    return records[:target]


def _parse_qa_pairs(text: str) -> List[Dict]:
    """
    Parse "Q: ... A: ..." formatted text into structured records.
    Handles minor formatting variations that the LLM sometimes produces.
    """
    pairs   = []
    lines   = text.strip().split("\n")
    current = {}

    for line in lines:
        line = line.strip()
        if line.lower().startswith("q:") or line.lower().startswith("question:"):
            current = {"instruction": re.sub(r"^q(?:uestion)?:\s*", "", line, flags=re.IGNORECASE).strip(),
                       "input": "", "output": ""}
        elif (line.lower().startswith("a:") or line.lower().startswith("answer:")) and current:
            current["output"] = re.sub(r"^a(?:nswer)?:\s*", "", line, flags=re.IGNORECASE).strip()
            if current["instruction"] and current["output"]:
                pairs.append(dict(current))
            current = {}

    return pairs


# ------------------------------------------------------------------
# Dataset saving.
# ------------------------------------------------------------------
def save_dataset(records: List[Dict], out_dir: Path) -> None:
    """
    Split records 90/10 into train and eval sets, then save as JSONL.
    JSONL is the expected input format for the Unsloth fine-tuning notebook.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(records)

    split     = int(len(records) * 0.9)
    train_set = records[:split]
    eval_set  = records[split:]

    for name, rows in [("train", train_set), ("eval", eval_set)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Saved {len(rows)} records -> {path}")

    # Also save the full dataset as a single JSON for inspection.
    all_path = out_dir / "dataset_full.json"
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Saved full dataset -> {all_path}")


# ------------------------------------------------------------------
# Entry point.
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning dataset.")
    parser.add_argument("--mode",   choices=["local", "groq"], default="local",
                        help="Generation mode.  'local' requires no API.  'groq' uses the free Groq API.")
    parser.add_argument("--target", type=int, default=300,
                        help="Target number of Q-A pairs to generate.")
    parser.add_argument("--data",   default="./data",
                        help="Path to documents folder (same as ingest.py).")
    parser.add_argument("--out",    default="./dataset",
                        help="Output directory for JSONL files.")
    args = parser.parse_args()

    print("\nCollegeBot Dataset Generator")
    print("=" * 50)
    print(f"  Mode   : {args.mode}")
    print(f"  Target : {args.target} Q-A pairs")
    print(f"  Source : {args.data}")

    # Load raw text from the same document set as ingest.py.
    from ingest import load_documents, split_documents
    data_dir = Path(args.data)
    docs     = load_documents(data_dir)
    chunks   = split_documents(docs)
    texts    = [c.page_content for c in chunks]

    if not texts:
        print("ERROR: No text extracted from data/.  Check your documents.")
        return

    print(f"\n  Source chunks available: {len(texts)}")

    if args.mode == "groq":
        records = generate_groq(texts, args.target)
    else:
        records = generate_local(texts, args.target)

    print(f"\n  Total Q-A pairs generated: {len(records)}")

    # Print a few samples for visual inspection.
    print("\n  Sample records:")
    print("  " + "-" * 58)
    for r in records[:3]:
        print(f"  Q: {r['instruction']}")
        print(f"  A: {r['output'][:100]}...")
        print()

    save_dataset(records, Path(args.out))
    print("\nDataset generation complete.")
    print(f"  Next step: upload dataset/ to Google Drive, then run the Colab notebook.")


if __name__ == "__main__":
    main()
