"""
ingest.py  --  Data Ingestion and FAISS Vector Store Builder
=============================================================
"Without a systematic way to start and keep data clean, bad data
 will happen."  -- Donato Diorio

ChromaDB was replaced with FAISS because chroma-hnswlib requires
Microsoft C++ Build Tools to compile on Windows.  FAISS ships a
pre-built Windows wheel and requires no compiler at all.

FAISS stores the index in two files:
    faiss_store/index.faiss   -- the binary vector index
    faiss_store/index.pkl     -- the document metadata store

Both files must be committed to git for Streamlit Cloud to work.

Usage:
    python ingest.py
    python ingest.py --sample
    python ingest.py --data ./my_docs --faiss ./my_faiss_store
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ------------------------------------------------------------------
# Configuration defaults
# ------------------------------------------------------------------
DEFAULT_DATA_DIR   = "./data"
DEFAULT_FAISS_PATH = "./faiss_store"
DEFAULT_EMBED_MODEL= "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 150
BATCH_SIZE         = 500


# ------------------------------------------------------------------
# Document loading
# ------------------------------------------------------------------
def load_documents(data_dir: Path) -> List[Document]:
    """
    Walk data_dir recursively and load all supported documents.
    Errors on individual files are caught without aborting the run.
    """
    from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader,
        UnstructuredHTMLLoader,
    )

    loader_map = {
        ".pdf" : PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt" : TextLoader,
        ".html": UnstructuredHTMLLoader,
    }

    all_docs: List[Document] = []
    found = list(data_dir.rglob("*.*"))
    print(f"\nScanning {data_dir} -- found {len(found)} files total.")

    for fpath in found:
        ext = fpath.suffix.lower()
        if ext not in loader_map:
            continue
        try:
            loader = loader_map[ext](str(fpath))
            docs   = loader.load()
            for doc in docs:
                doc.metadata["source"]    = str(fpath)
                doc.metadata["file_name"] = fpath.name
                doc.metadata["file_type"] = ext
            all_docs.extend(docs)
            print(f"  Loaded  {fpath.name}  ({len(docs)} section(s))")
        except Exception as e:
            print(f"  SKIP    {fpath.name}  -- {e}")

    print(f"\nTotal documents loaded: {len(all_docs)}")
    return all_docs


# ------------------------------------------------------------------
# Text splitting
# ------------------------------------------------------------------
def split_documents(
    docs: List[Document],
    chunk_size: int   = CHUNK_SIZE,
    chunk_overlap: int= CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks.
    Overlap ensures sentences straddling chunk boundaries are
    fully represented in at least one adjacent chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = chunk_size,
        chunk_overlap   = chunk_overlap,
        separators      = ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function = len,
    )
    chunks = splitter.split_documents(docs)
    print(f"\nChunks after splitting: {len(chunks)}")
    return chunks


# ------------------------------------------------------------------
# FAISS vector store construction
# ------------------------------------------------------------------
def build_vectorstore(
    chunks    : List[Document],
    faiss_path: str = DEFAULT_FAISS_PATH,
    embed_model: str= DEFAULT_EMBED_MODEL,
) -> FAISS:
    """
    Embed chunks with all-MiniLM-L6-v2 and save a FAISS index.

    FAISS does not require a server or C++ compiler at runtime.
    The index is saved as two files that can be committed to git
    and loaded instantly on Streamlit Cloud.
    """
    print(f"\nLoading embedding model: {embed_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name    = embed_model,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )
    print("Embedding model loaded.  Vector dimension: 384")

    print(f"\nBuilding FAISS index...")
    t0 = time.time()

    # Build the FAISS index from the first batch.
    vectorstore = FAISS.from_documents(chunks[:BATCH_SIZE], embeddings)

    # Add remaining batches incrementally.
    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch     = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Adding batch {batch_num} ({len(batch)} chunks)...")
        vectorstore.add_documents(batch)

    # Persist to disk.
    os.makedirs(faiss_path, exist_ok=True)
    vectorstore.save_local(faiss_path)
    elapsed = round(time.time() - t0, 1)
    print(f"\nFAISS index saved to {faiss_path}  ({elapsed}s)")
    return vectorstore


# ------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------
def verify(vectorstore: FAISS) -> None:
    """
    Run a quick MMR retrieval smoke test to confirm the index works.
    """
    print("\nVerification -- MMR retrieval smoke test:")
    results = vectorstore.max_marginal_relevance_search(
        "What are the admission requirements?",
        k=3, fetch_k=10,
    )
    if not results:
        print("  WARNING: No results returned.  Check your documents.")
        return
    for i, doc in enumerate(results, 1):
        preview = doc.page_content[:100].replace("\n", " ").strip()
        src     = doc.metadata.get("file_name", "?")
        print(f"  [{i}] ({src}) {preview}...")
    print("  Smoke test passed.")


# ------------------------------------------------------------------
# Sample data generator
# ------------------------------------------------------------------
def create_sample_data(data_dir: Path) -> None:
    """
    Write a minimal college handbook to data_dir for testing.
    Replace with real documents before going to production.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    sample = data_dir / "sample_college_handbook.txt"
    if sample.exists():
        print(f"Sample data already exists at {sample}")
        return

    text = """\
COLLEGE STUDENT HANDBOOK 2024-25
=================================

ADMISSION REQUIREMENTS
The last date to submit the college admission form is June 30, 2024.
Required documents for admission:
  - 10th and 12th mark sheets (originals and two photocopies each)
  - Transfer Certificate (TC) from the previous institution
  - Migration Certificate (for students from other universities)
  - Character Certificate
  - Four passport-size photographs
  - Aadhar Card or equivalent government-issued ID
  - Category Certificate (SC/ST/OBC/EWS, if applicable)

FEE STRUCTURE 2024-25
Annual fee for B.Tech: Rs 85,000
  Breakdown: Tuition Rs 65,000 + Development Rs 12,000 + Examination Rs 8,000
Annual fee for B.Com: Rs 42,000
Annual fee for BCA: Rs 48,000
Annual fee for MBA: Rs 95,000
Fees are payable in two equal installments:
  First installment: on or before July 15
  Second installment: on or before January 15

SCHOLARSHIPS
1. Chief Minister Scholarship -- for students with family income below Rs 2.5 lakh per annum.
2. Merit-cum-Means Scholarship -- for top 10 percent students with family income below Rs 4.5 lakh per annum.
3. College Merit Scholarship -- for students scoring above 90 percent in 12th board examinations.
4. Sports Scholarship -- for state-level and national-level sports achievers.
Applications for all scholarships open from August 1 to September 30 each year.

ATTENDANCE POLICY
Students must maintain a minimum of 75 percent attendance in each subject to be
eligible to appear in semester examinations.
Students with attendance between 65 and 74 percent may apply for condonation
to the Principal with a penalty of Rs 500 per subject.
Students below 65 percent attendance will be detained and cannot appear in exams.

COURSE REGISTRATION
Students register for elective courses through the online ERP portal at erp.college.edu.
The course registration window opens in Week 3 of each semester, Monday to Friday, 9 AM to 5 PM.
Late registration incurs a penalty of Rs 200 per day, up to a maximum of five days.
After five days, registration closes and the student must wait for the next semester.

HOSTEL ACCOMMODATION
Students applying for hostel must submit the Hostel Application Form to the
Office of the Dean of Student Welfare before the start of each semester.
Hostel fee: Rs 48,000 per annum (includes room, electricity, and Wi-Fi).
Mess fee: Rs 36,000 per annum (vegetarian and non-vegetarian options available).
A refundable security deposit of Rs 5,000 is collected at the time of admission.

LIBRARY
The Central Library is located in Block C, Ground Floor.
Working hours: Monday to Saturday, 8:00 AM to 8:00 PM.
Sunday: 10:00 AM to 4:00 PM.
Students may borrow up to four books at a time for a period of 14 days.
Overdue fine: Rs 2 per book per day.

EXAMINATION SCHEDULE
End-semester examinations are held in November-December for odd semesters
and in April-May for even semesters.
Mid-semester examinations are held in September and February respectively.
Date sheets are published on the official college website 30 days before exams.
Students must carry their hall ticket to every examination hall.

GRIEVANCE REDRESSAL
Students may submit academic grievances within 15 days of result declaration.
Submit grievance forms to the Academic Section, Administrative Block, Room 102.
Response time: 10 working days.
For urgent issues, contact the Student Welfare Officer at welfare@college.edu.
"""
    sample.write_text(text, encoding="utf-8")
    print(f"Sample data created at {sample}")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build CollegeBot FAISS vector store.")
    p.add_argument("--data",   default=DEFAULT_DATA_DIR,   help="Path to documents folder.")
    p.add_argument("--faiss",  default=DEFAULT_FAISS_PATH, help="Path to FAISS output folder.")
    p.add_argument("--model",  default=DEFAULT_EMBED_MODEL,help="Embedding model name.")
    p.add_argument("--sample", action="store_true",        help="Create sample data if data/ is empty.")
    return p.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    data_dir = Path(args.data)

    print("CollegeBot -- Data Ingestion Pipeline (FAISS)")
    print("=" * 50)

    if args.sample or not any(data_dir.rglob("*.*")):
        print("Creating sample data...")
        create_sample_data(data_dir)

    docs   = load_documents(data_dir)
    chunks = split_documents(docs)

    if not chunks:
        print("\nERROR: No text extracted.  Check that data/ contains readable files.")
        sys.exit(1)

    vs = build_vectorstore(chunks, args.faiss, args.model)
    verify(vs)

    print("\nIngestion complete.")
    print(f"  Documents : {len(docs)}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  FAISS DB  : {args.faiss}")
    print("\nNext step: streamlit run app.py")
